from math import sqrt
import numpy as np
from ..traitlets import Float, Int, Instance, Enum, Unicode, Bool, HasTraits, List
from ..util import limit, Afgen, merge_dict, DotMap

from .. import exceptions as exc


class pFCurve(Afgen):
    """Pf curve should check that:
    - length of array is even
    - has at least 3 xy pairs
    - pF should start at pF = -1 and end at pF=6
    - SM/cond values should decrease with increase pF

    """
    pass

class MFPCurve(Afgen):
    """Computes Matrix Flow Potential using a Gaussian integration over the pfCurve

    """
    elog10 = 2.302585092994
    Pgauss = (0.0469100770, 0.2307653449, 0.5000000000, 0.7692346551, 0.9530899230)
    Wgauss = (0.1184634425, 0.2393143352, 0.2844444444, 0.2393143352, 0.1184634425)

    def __init__(self, SMfromPF, CONDfromPF):
        SMfromPF = np.array(SMfromPF)
        CONDfromPF = pFCurve(CONDfromPF)
        MFPfromPF = np.zeros_like(SMfromPF)

        # zero MFP at highest pF
        MFPfromPF[-1] = 0.
        MFPfromPF[-2] = SMfromPF[-2]

        for ip in range(len(SMfromPF) - 3, 0, -2):
            # Copy corresponding pF value
            MFPfromPF[ip - 1] = SMfromPF[ip - 1]
            # get integral over PF range
            add = 0.0
            DeltaPF = SMfromPF[ip + 1] - SMfromPF[ip - 1]
            for i in range(len(self.Pgauss)):
                PFg = SMfromPF[ip - 1] + self.Pgauss[i] * DeltaPF
                CON = 10.0 ** CONDfromPF(PFg)
                add += CON * 10.0 ** PFg * self.elog10 * self.Wgauss[i]
            MFPfromPF[ip] = add * DeltaPF + MFPfromPF[ip + 2]
        Afgen.__init__(self, MFPfromPF)

class HeightFromAirCurve(Afgen):
    "Computes the pressure head from the amount of air in a layer"
    
    def __init__(self, SMfromPF, WaterFromHeight):
        SMsat = SMfromPF(-1.0)
        HeightFromAir = [None] * len(WaterFromHeight.x_list) * 2
        
        for i in range(0, len(WaterFromHeight.x_list)):
            HeightFromAir[2*i] = WaterFromHeight.x_list[i] * SMsat - WaterFromHeight.y_list[i]
            HeightFromAir[2*i+1] = WaterFromHeight.x_list[i]            
        Afgen.__init__(self, HeightFromAir) 

class WaterFromHeightCurve(Afgen):
    "Computes the amount of water in a layer from the matrix head"
    
    def __init__(self, SMfromPF):
        MH0 = 0.0
        MH1 = 2.0
        # Pgauss = [0.1127016654,.5,0.8872983346]
        # Wgauss = [0.2777778,0.4444444,0.2777778]
        Pgauss = [0.0469100770, 0.2307653449, 0.5000000000, 0.7692346551, 0.9530899230]
        Wgauss = [0.1184634425, 0.2393143352, 0.2844444444, 0.2393143352, 0.1184634425]
        Ngauss = len(Pgauss)
        EquilTableLen = 30

        WaterFromHeight = [None] * EquilTableLen
        WaterFromHeight[0] = 0.0
        WaterFromHeight[1] = 0.0

        for i in range(2,int((EquilTableLen/2)) + 1):
            WaterFromHeight[2*i-2] = MH1
            WaterFromHeight[2*i-1] = WaterFromHeight[2*i-3]
            for j in range(1,Ngauss+1):
                PFg = np.log10(MH0 + Pgauss[j-1] * (MH1-MH0))
                WaterFromHeight[2*i-1] = WaterFromHeight[2*i-1] + Wgauss[j-1] * (MH1-MH0) * SMfromPF(PFg)
            MH0 = MH1
            MH1 = 2*MH1
        Afgen.__init__(self, WaterFromHeight)

class SoilLayer(HasTraits):
    """Contains the intrinsic and derived properties for each soil layers
    """
    SMfromPF = Instance(pFCurve)  # soil moisture content as function of pF
    CONDfromPF = Instance(pFCurve)  # conductivity as function of pF
    PFfromSM = Instance(Afgen)  # Inverse from SMfromPF
    MFPfromPF = Instance(MFPCurve)  # Matrix Flux Potential as function of pF
    CNRatioSOMI = Float()  # Initial C:N ratio of soil organic matter
    FSOMI = Float()  # Initial fraction of soil organic matter in soil
    RHOD = Float()  # Bulk density of the soil
    Soil_pH = Float()  # pH of the soil layer
    CRAIRC = Float()  # Critical air content
    SMsat = Float()  # Saturation soil moisture content
    DepthLowerBoundary = Float()
    Thickness = Float()
    rooting_status = Enum(["rooted","partially rooted","potentially rooted","never rooted",])

    def __init__(self, layer, PFFieldCapacity, PFWiltingPoint):
        self.SMfromPF = pFCurve(layer.SMfromPF)
        self.CONDfromPF = pFCurve(layer.CONDfromPF)
        self.PFfromSM = self._invert_pF(layer.SMfromPF)
        self.MFPfromPF = MFPCurve(layer.SMfromPF, layer.CONDfromPF)
        self.CNRatioSOMI = layer.CNRatioSOMI
        self.FSOMI = layer.FSOMI
        self.RHOD = layer.RHOD
        self.CRAIRC = layer.CRAIRC
        self.Soil_pH = layer.Soil_pH      

        # if 5 <= layer.Thickness <= 250:
        #     self.Thickness = layer.Thickness
        # else:
        #     msg = "Soil layer should have thickness between 5 and 250 cm. Current value: %f" % layer.Thickness
        #     raise exc.PCSEError(msg)
        
        self.Thickness = layer.Thickness

        self.CRAIRC = layer.CRAIRC
        self.SM0 = self.SMfromPF(-1.0)
        self.SMFCF = self.SMfromPF(PFFieldCapacity)
        self.SMW = self.SMfromPF(PFWiltingPoint)

        self.WC0 = self.SM0 * self.Thickness
        self.WCW = self.SMW * self.Thickness
        self.WCFC = self.SMFCF * self.Thickness
        self.CondFC = 10.0 ** self.CONDfromPF(PFFieldCapacity)
        self.CondK0 = 10.0 ** self.CONDfromPF(-1.0)
        self.rooting_status = None

        # compute hash value of this layer
        self._hash = hash((tuple(layer.SMfromPF), tuple(layer.CONDfromPF)))
        
    @property
    def Thickness_m(self):
        return self.Thickness * 1e-2

    @property
    def RHOD_kg_per_m3(self):
        return self.RHOD * 1e-3 * 1e06

    def _invert_pF(self, SMfromPF):
        """Inverts the SMfromPF table to get pF from SM
        """
        l = []
        for t in zip(reversed(SMfromPF[1::2]), reversed(SMfromPF[0::2])):
            l.extend(t)
        return Afgen(l)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self._hash == other._hash
        )


class SoilProfile(list):
    """A component that represents the soil column as required by the multilayer waterbalance and SNOMIN.


    """
    
    def __init__(self, parvalues):
        list.__init__(self)

        sp = DotMap(parvalues["SoilProfileDescription"])
        for layer_properties in sp.SoilLayers:
            layer = SoilLayer(layer_properties, sp.PFFieldCapacity, sp.PFWiltingPoint)
            self.append(layer)
        for attr, value in sp.items():
            if attr == "SoilLayers":
                continue
            if attr == "SubSoilType":
                value = SoilLayer(value, sp.PFFieldCapacity, sp.PFWiltingPoint)
            setattr(self, attr, value)
        for layer in sp.SoilLayers:
            if 5 <= layer.Thickness <= 250:
                continue
            else:
                msg = "Soil layer should have thickness between 5 and 250 cm. Current value: %f" % layer.Thickness
                raise exc.PCSEError(msg)        
            
            
    def determine_depth_lower_boundaries(self):
        depth = 0.
        for layer in self:
            depth += layer.Thickness
            layer.DepthLowerBoundary = depth
        self.SubSoilType.DepthLowerBoundary = self.SubSoilType.Thickness + depth

    def determine_rooting_status(self, RD, RDM):
        """Determines the rooting status of the soil layers and update layer weights.

        Soil layers can be rooted, partially rooted, potentially rooted or never rooted.
        This is stored in layer.rooting_status

        Note that this routine expected that the maximum rooting depth coincides
        with a layer boundary.

        :param RD:the current rooting depth
        :param RDM: the maximum rooting depth
        """
        upper_layer_boundary = 0
        lower_layer_boundary = 0
        for layer in self:
            lower_layer_boundary += layer.Thickness
            if lower_layer_boundary <= RD:
                layer.rooting_status = "rooted"
            elif upper_layer_boundary < RD < lower_layer_boundary:
                layer.rooting_status = "partially rooted"
            elif RD < lower_layer_boundary <= RDM:
                layer.rooting_status = "potentially rooted"
            else:
                layer.rooting_status = "never rooted"
            upper_layer_boundary = lower_layer_boundary

        self._compute_layer_weights(RD)

    def _compute_layer_weights(self, RD):
        """computes the layer weights given the current rooting depth.

        :param RD: The current rooting depth
        """
        lower_layer_boundary = 0
        for layer in self:
            lower_layer_boundary += layer.Thickness
            if layer.rooting_status == "rooted":
                layer.Wtop = 1.0
                layer.Wpot = 0.0
                layer.Wund = 0.0
            elif layer.rooting_status == "partially rooted":
                layer.Wtop = 1.0 - (lower_layer_boundary - RD) / layer.Thickness
                layer.Wpot = 1.0 - layer.Wtop
                layer.Wund = 0.0
            elif layer.rooting_status == "potentially rooted":
                layer.Wtop = 0.0
                layer.Wpot = 1.0
                layer.Wund = 0.0
            elif layer.rooting_status == "never rooted":
                layer.Wtop = 0.0
                layer.Wpot = 0.0
                layer.Wund = 1.0
            else:
                msg = "Unknown rooting status: %s" % layer.rooting_status
                raise exc.PCSEError(msg)

    def validate_max_rooting_depth(self, RDM):
        """Validate that the maximum rooting depth coincides with a layer boundary.

        :param RDM: The maximum rootable depth
        :return: True or False
        """
        tiny = 0.01
        lower_layer_boundary = 0
        for layer in self:
            lower_layer_boundary += layer.Thickness
            if abs(RDM - lower_layer_boundary) < tiny:
                break
        else:  # no break
            msg = "Current maximum rooting depth (%f) does not coincide with a layer boundary!" % RDM
            raise exc.PCSEError(msg)

    def get_max_rootable_depth(self):
        """Returns the maximum soil rootable depth.

        here we assume that the max rootable depth is equal to the lower boundary of the last layer.

        :return: the max rootable depth in cm
        """
        LayerThickness = [l.Thickness for l in self]
        LayerLowerBoundary = list(np.cumsum(LayerThickness))
        return max(LayerLowerBoundary)