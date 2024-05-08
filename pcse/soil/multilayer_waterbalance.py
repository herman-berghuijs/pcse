from math import sqrt
import numpy as np

from ..traitlets import Float, Int, Instance, Bool
from ..decorators import prepare_rates, prepare_states
from ..util import limit
from ..base import ParamTemplate, StatesTemplate, RatesTemplate, \
     SimulationObject
from .. import exceptions as exc
from .. import signals

from .soil_profile import SoilProfile

REFERENCE_TEST_RUN = True  # set by testing procedure    

class WaterBalanceLayered_PP(SimulationObject):
    _default_RD = Float(10.)  # default rooting depth at 10 cm
    _RDold = _default_RD
    _RDM = Float(None)
    
    # Counter for Days-Dince-Last-Rain
    DSLR = Float(1)
    
    # rainfall rate of previous day
    RAINold = Float(0)

    # placeholders for soil object and parameter provider
    soil_profile = None

    # Indicates that a new crop has started
    crop_start = Bool(False)

    class Parameters(ParamTemplate):
        pass

    class StateVariables(StatesTemplate):
        SM = Instance(np.ndarray)
        WC = Instance(np.ndarray)
        WTRAT = Float(-99.)
        EVST = Float(-99.)

    class RateVariables(RatesTemplate):
        EVS = Float(-99.)
        WTRA = Float(-99.)

    def initialize(self, day, kiosk, parvalues):
        self.soil_profile = SoilProfile(parvalues)
        parvalues._soildata["soil_profile"] = self.soil_profile

        # Maximum rootable depth
        self._RDM = self.soil_profile.get_max_rootable_depth()
        self.soil_profile.validate_max_rooting_depth(self._RDM)

        SM = np.zeros(len(self.soil_profile))
        WC = np.zeros_like(SM)
        for il, layer in enumerate(self.soil_profile):
            SM[il] = layer.SMFCF
            WC[il] = SM[il] * layer.Thickness
        
        WTRAT = 0.
        EVST = 0.

        states = { "WC": WC, "SM":SM, "EVST": EVST, "WTRAT": WTRAT}
        self.rates = self.RateVariables(kiosk, publish="EVS")      
        self.states = self.StateVariables(kiosk, publish=["WC", "SM", "EVST"], **states)

    @prepare_rates
    def calc_rates(self, day, drv):
        r = self.rates
        # Transpiration and maximum soil and surface water evaporation rates
        # are calculated by the crop Evapotranspiration module.
        # However, if the crop is not yet emerged then set TRA=0 and use
        # the potential soil/water evaporation rates directly because there is
        # no shading by the canopy.
        if "TRA" not in self.kiosk:
            r.WTRA = 0.
            EVSMX = drv.ES0
        else:
            r.WTRA = self.kiosk["TRA"]
            EVSMX = self.kiosk["EVSMX"]

        # Actual evaporation rates

        if self.RAINold >= 1:
            # If rainfall amount >= 1cm on previous day assume maximum soil
            # evaporation
            r.EVS = EVSMX
            self.DSLR = 1.
        else:
            # Else soil evaporation is a function days-since-last-rain (DSLR)
            self.DSLR += 1
            EVSMXT = EVSMX * (sqrt(self.DSLR) - sqrt(self.DSLR - 1))
            r.EVS = min(EVSMX, EVSMXT + self.RAINold)

        # Hold rainfall amount to keep track of soil surface wetness and reset self.DSLR if needed
        self.RAINold = drv.RAIN
        
    @prepare_states
    def integrate(self, day, delt=1.0):
        self.states.SM = self.states.SM
        self.states.WC = self.states.WC

        # Accumulated transpiration and soil evaporation amounts
        self.states.EVST += self.rates.EVS * delt
        self.states.WTRAT += self.rates.WTRA * delt
        

class WaterBalanceLayered(SimulationObject):
    """This implements a layered water balance to estimate soil water availability for crop growth and water stress.

    The classic free-drainage water-balance had some important limitations such as the inability to take into
    account differences in soil texture throughout the profile and its impact on soil water flow. Moreover,
    in the single layer water balance, rainfall or irrigation will become immediately available to the crop.
    This is incorrect physical behaviour and in many situations it leads to a very quick recovery of the crop
    after rainfall since all the roots have immediate access to infiltrating water. Therefore, with more detailed
    soil data becoming available a more realistic soil water balance was deemed necessary to better simulate soil
    processes and its impact on crop growth.

    The multi-layer water balance represents a compromise between computational complexity, realistic simulation
    of water content and availability of data to calibrate such models. The model still runs on a daily time step
    but does implement the concept of downward and upward flow based on the concept of hydraulic head and soil
    water conductivity. The latter are combined in the so-called Matric Flux Potential. The model computes
    two types of flow of water in the soil:

      (1) a "dry flow" from the matric flux potentials (e.g. the suction gradient between layers)
      (2) a "wet flow" under the current layer conductivities and downward gravity.

    Clearly, only the dry flow may be negative (=upward). The dry flow accounts for the large
    gradient in water potential under dry conditions (but neglects gravity). The wet flow takes into
    account gravity only and will dominate under wet conditions. The maximum of the dry and wet
    flow is taken as the downward flow, which is then further limited in order the prevent
    (a) oversaturation and (b) water content to decrease below field capacity.
    Upward flow is just the dry flow when it is negative. In this case the flow is limited
    to a certain fraction of what is required to get the layers at equal potential, taking
    into account, however, the contribution of an upward flow from further down.

    The configuration of the soil layers is variable but is bound to certain limitations:
    - The layer thickness cannot be made too small. In practice, the top layer should not
      be smaller than 20 to 30 cm. Smaller layers would require smaller time steps than
      one day to simulate realistically, since rain storms will fill up the top layer very
      quickly leading to surface runoff because the model cannot handle the infiltration of
      the rainfall in a single timestep (a day).
    - The crop maximum rootable depth must coincide with a layer boundary. This is to avoid
      that roots can directly access water below the rooting depth. Of course such water may become
      available gradually by upward flow of moisture at some point during the simulation.

    The current python implementation does not yet implement the impact of shallow groundwater
    but this will be added in future versions of the model.

    **note**: the current implementation of the model is rather 'Fortran-ish'. This has been done
    on purpose to allow comparisons with the original code in Fortran90. When we are sure that
    the implementation performs properly, we can refactor this in to a more functional structure
    instead of the

    """
    
    
    _default_RD = Float(10.)  # default rooting depth at 10 cm
    _RDold = _default_RD
    _RINold = Float(0.)
    _RIRR = Float(0.)
    _DSLR = Int(None)
    _RDM = Float(None)
    _RAIN = Float(None)
    _WCI = Float(None)

    # Max number of flow iterations and precision required
    MaxFlowIter = 50
    TinyFlow = 0.001

    # Maximum upward flow is 50% of amount needed to reach equilibrium between layers
    # see documentation Kees Rappoldt - page 80
    UpwardFlowLimit = 0.50
    
    # Maximum depth that the groundwater level can reach
    XDEF = 1000.

    # placeholders for soil object and parameter provider
    soil_profile = None
    parameter_provider = None

    # Indicates that a new crop has started
    crop_start = Bool(False)

    class Parameters(ParamTemplate):
        AUTOIRR = Bool(False)
        IRR_threshold = Float(0.)

        IFUNRN = Int(None)
        NOTINF = Float(None)
        SSI = Float(None)
        SSMAX = Float(None)
        SMLIM = Float(None)
        WAV = Float(None)
        
        IDRAIN = Bool(None)
        DD = Float(None)
        ZTI = Float(None)

    class StateVariables(StatesTemplate):
        WTRAT = Float(None)
        EVST = Float(None)
        EVWT = Float(None)
        TSR = Float(None)
        RAINT = Float(None)
        WDRT = Float(None)
        TOTINF = Float(None)
        TOTIRR = Float(None)
        CRT = Float(None)
        SM = Instance(np.ndarray)
        SM_MEAN = Float(None)
        WC = Instance(np.ndarray)
        W = Float(None)
        WLOW = Float(None)
        WWLOW = Float(None)
        WBOT = Float(None)
        WAVUPP = Float(None)
        WAVLOW = Float(None)
        WAVBOT = Float(None)
        SS = Float(None)
        BOTTOMFLOWT = Float(None) 
        
        WSUB = Float(None)
        ZT = Float(None)


    class RateVariables(RatesTemplate):
        Flow = Instance(np.ndarray)
        RIN = Float(None)
        WTRALY = Instance(np.ndarray)
        WTRA = Float(None)
        EVS = Float(None)
        EVW = Float(None)
        RIRR = Float(None)
        DWC = Instance(np.ndarray)
        DRAINT = Float(None)
        DSS = Float(None)
        DTSR = Float(None)
        BOTTOMFLOW = Float(None)        
        DWSUB = Float(None)

    def initialize(self, day, kiosk, parvalues):
        from .soil_profile import SoilProfile, WaterFromHeightCurve, HeightFromAirCurve
        
        self.soil_profile = SoilProfile(parvalues) 
        self.soil_profile.determine_depth_lower_boundaries()     
        if(self.soil_profile.GroundWater):                   
            for il, layer in enumerate(self.soil_profile):
                layer.WaterFromHeight = WaterFromHeightCurve(layer.SMfromPF)
                layer.HeightFromAir = HeightFromAirCurve(layer.SMfromPF, layer.WaterFromHeight)
            self.soil_profile.SubSoilType.WaterFromHeight = self.soil_profile[-1].WaterFromHeight
            self.soil_profile.SubSoilType.HeightFromAir = self.soil_profile[-1].HeightFromAir                

        parvalues._soildata["soil_profile"] = self.soil_profile

        # Maximum rootable depth
        RDMsoil = self.soil_profile.get_max_rootable_depth()
        if REFERENCE_TEST_RUN:
            # Crop rooting depth (RDMCR) is required at start for comparison with
            # results from fortran code
            self._RDM = min(parvalues["RDMCR"], RDMsoil)
        else:
            self._RDM = self.soil_profile.get_max_rootable_depth()
        self.soil_profile.validate_max_rooting_depth(self._RDM)

        self.params = self.Parameters(parvalues)
        p = self.params

        # Store the parameterprovider because we need it to retrieve the maximum
        # crop rooting depth when a new crop is started.
        self.parameter_provider = parvalues

        self.soil_profile.determine_rooting_status(self._default_RD, self._RDM)

        SM = np.zeros(len(self.soil_profile))
        WC = np.zeros_like(SM)
        Flow = np.zeros(len(self.soil_profile) + 1)
        
        if self.soil_profile.GroundWater:
            RD = self._determine_rooting_depth()
            ZT = limit(0.1, self.soil_profile.SubSoilType.DepthLowerBoundary, p.ZTI)

            if(p.IDRAIN):
                    ZT = max(ZT, p.DD)  # corrected for drainage depth
            else:
                pass
            
            for il, layer in enumerate(self.soil_profile):
                LBSL = layer.DepthLowerBoundary
                TSL = layer.Thickness    
                SM0 = layer.SM0
                SMfromPF = layer.SMfromPF
                WaterFromHeight = layer.WaterFromHeight
                HH = LBSL - TSL / 2.0 # depth at half-layer-height                
                if (LBSL - ZT < 0.0):
                    # layer is above groundwater ; get equilibrium amount from Half-Height pressure head
                    SM[il] = layer.SMfromPF(np.log10(ZT-HH)) 
                elif(LBSL - TSL >= ZT):
                    # layer completely in groundwater
                    SM[il] = SM0
                else:
                    # layer partly in groundwater
                    SM[il] = ((LBSL-ZT)*SM0 + WaterFromHeight(ZT-(LBSL-TSL))) / TSL
                WC[il] = SM[il] * TSL
            #    calculate (available) water in rooted and potentially rooted zone
            #    note that water amount WBOT below depth RDM (RDMSLB)
            #    is never available ; it is below potential rooting depth                    
            W    = 0.0
            WAVUPP = 0.0
            WLOW = 0.0
            WAVLOW = 0.0
            WBOT = 0.0
            WAVBOT = 0.0 
            
            for il, layer in enumerate(self.soil_profile):
                W      = W    + SM[il] * layer.Thickness * layer.Wtop
                WLOW   = WLOW + SM[il] * layer.Thickness * layer.Wpot
                WBOT   = WBOT + SM[il] * layer.Thickness * layer.Wund
            #  available water
                WAVUPP = WAVUPP + (SM[il]-layer.SMW) * layer.Thickness * layer.Wtop
                WAVLOW = WAVLOW + (SM[il]-layer.SMW) * layer.Thickness * layer.Wpot

            # now various subsoil amonts 
            LBSLBOT = self.soil_profile[-1].DepthLowerBoundary
            LBSUB = self.soil_profile.SubSoilType.DepthLowerBoundary
            SubSM0 = self.soil_profile.SubSoilType.SM0
            WaterFromHeight = self.soil_profile.SubSoilType.WaterFromHeight
            
            if(ZT > LBSLBOT):
                # Groundwater below layered system
                WSUB =  (LBSUB - ZT) * SubSM0 + WaterFromHeight(ZT - LBSLBOT)
            else:
                # Saturated sub soil
                WSUB = (LBSUB - LBSLBOT) * SubSM0
            
            # Then amount of moisture below rooted zone
            WZ = WLOW + WBOT + WSUB
            WZI = WZ
        else:
            # AVMAX -  maximum available content of layer(s)
            # This is calculated first to achieve an even distribution of water in the rooted top
            # if WAV is small. Note the separate limit for initial SM in the rooted zone.
            TOPLIM = 0.0
            LOWLIM = 0.0
            AVMAX = []
            for il, layer in enumerate(self.soil_profile):
                if layer.rooting_status in ["rooted", "partially rooted"]:
                    # Check whether SMLIM is within boundaries
                    SML = limit(layer.SMW, layer.SM0, p.SMLIM)
                    AVMAX.append((SML - layer.SMW) * layer.Thickness)   # available in cm
                    # also if partly rooted, the total layer capacity counts in TOPLIM
                    # this means the water content of layer ILR is set as if it would be
                    # completely rooted. This water will become available after a little
                    # root growth and through numerical mixing each time step.
                    TOPLIM += AVMAX[il]
                elif layer.rooting_status == "potentially rooted":
                    # below the rooted zone the maximum is saturation (see code for WLOW in one-layer model)
                    # again the full layer capacity adds to LOWLIM.
                    SML = layer.SM0
                    AVMAX.append((SML - layer.SMW) * layer.Thickness)   # available in cm
                    LOWLIM += AVMAX[il]
                else:  # Below the potentially rooted zone
                    break

            if p.WAV <= 0.0:
                # no available water
                TOPRED = 0.0
                LOWRED = 0.0
            elif p.WAV <= TOPLIM:
                # available water fits in layer(s) 1..ILR, these layers are rooted or almost rooted
                # reduce amounts with ratio WAV / TOPLIM
                TOPRED = p.WAV / TOPLIM
                LOWRED = 0.0
            elif p.WAV < TOPLIM + LOWLIM:
                # available water fits in potentially rooted layer
                # rooted zone is filled at capacity ; the rest reduced
                TOPRED = 1.0
                LOWRED = (p.WAV-TOPLIM) / LOWLIM
            else:
                # water does not fit ; all layers "full"
                TOPRED = 1.0
                LOWRED = 1.0

            W = 0.0    ; WAVUPP = 0.0
            WLOW = 0.0 ; WAVLOW = 0.0
            # SM = np.zeros(len(self.soil_profile))
            for il, layer in enumerate(self.soil_profile):
                if layer.rooting_status in ["rooted", "partially rooted"]:
                    # Part of the water assigned to ILR may not actually be in the rooted zone, but it will
                    # be available shortly through root growth (and through numerical mixing).
                    SM[il] = layer.SMW + AVMAX[il] * TOPRED / layer.Thickness
                    W += SM[il] * layer.Thickness * layer.Wtop
                    WLOW += SM[il] * layer.Thickness * layer.Wpot
                    # available water
                    WAVUPP += (SM[il] - layer.SMW) * layer.Thickness * layer.Wtop
                    WAVLOW += (SM[il] - layer.SMW) * layer.Thickness * layer.Wpot
                elif layer.rooting_status == "potentially rooted":
                    SM[il] = layer.SMW + AVMAX[il] * LOWRED / layer.Thickness
                    WLOW += SM[il] * layer.Thickness * layer.Wpot
                    # available water
                    WAVLOW += (SM[il] - layer.SMW) * layer.Thickness * layer.Wpot
                else:
                    # below the maximum rooting depth, set SM content to wilting point
                    SM[il] = layer.SMW
                WC[il] = SM[il] * layer.Thickness

                # set groundwater depth far away for clarity ; this prevents also
                # the root routine to stop root growth when they reach the groundwater
                ZT = 999.0
                WSUB = 0.

        # soil evaporation, days since last rain
        top_layer = self.soil_profile[0]
        top_layer_half_wet = top_layer.SMW + 0.5 * (top_layer.SMFCF - top_layer.SMW)
        self._DSLR = 5 if SM[0] <= top_layer_half_wet else 1

        # all summation variables of the water balance are set at zero.               
        states = {"WTRAT": 0., "EVST": 0., "EVWT": 0., "TSR": 0., "WDRT": 0.,
                    "TOTINF": 0., "TOTIRR": 0., "BOTTOMFLOWT": 0.,
                    "CRT": 0., "RAINT": 0., "WLOW": WLOW, "W": W, "WC": WC, "SM":SM,
                    "SS": p.SSI, "WWLOW": W+WLOW, "WBOT":0., "SM_MEAN": W/self._default_RD,
                    "WAVUPP": WAVUPP, "WAVLOW": WAVLOW, "WAVBOT":0., "WSUB": WSUB, "ZT": ZT
                    }
        self.states = self.StateVariables(kiosk, publish=["WC", "SM", "EVST"], **states)           

        # Initial values for profile water content
        self._WCI = WC.sum()

        # rate variables
        self.rates = self.RateVariables(kiosk, publish=["RIN", "Flow", "EVS"])
        self.rates.Flow = Flow


        # Connect to CROP_START/CROP_FINISH/IRRIGATE signals
        self._connect_signal(self._on_CROP_START, signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signals.crop_finish)
        self._connect_signal(self._on_IRRIGATE, signals.irrigate)


    @prepare_rates
    def calc_rates(self, day, drv):
        p = self.params
        s = self.states
        k = self.kiosk
        r = self.rates

        delt = 1.0

        # Update rooting setup if a new crop has started
        if self.crop_start:
            self.crop_start = False
            self._setup_new_crop()

        # Rate of irrigation (RIRR)
        r.RIRR = self._RIRR
        if "TRALY" in self.kiosk:
            if p.AUTOIRR:
                for il, layer in enumerate(self.soil_profile):
                    if(s.WC[il] < layer.WCFC):
                        r.RIRR+= layer.WCFC - s.WC[il]
                    else:
                        r.RIRR+= 0.
        self._RIRR = 0.       

        # copy rainfall rate for totalling in RAINT
        self._RAIN = drv.RAIN

        # Crop transpiration and maximum evaporation rates
        if "TRALY" in self.kiosk:
            # Transpiration and maximum soil and surface water evaporation rates
            # are calculated by the crop evapotranspiration module and taken from kiosk.
            WTRALY = k.TRALY
            r.WTRA = k.TRA
            EVWMX = k.EVWMX
            EVSMX = k.EVSMX
        else:
            # However, if the crop is not yet emerged then set WTRALY/TRA=0 and use
            # the potential soil/water evaporation rates directly because there is
            # no shading by the canopy.
            WTRALY = np.zeros_like(s.SM)
            r.WTRA = 0.
            EVWMX = drv.E0
            EVSMX = drv.ES0

        # Actual evaporation rates
        r.EVW = 0.
        r.EVS = 0.
        if s.SS > 1.:
            # If surface storage > 1cm then evaporate from water layer on soil surface
            r.EVW = EVWMX
        else:
            # else assume evaporation from soil surface
            if self._RINold >= 1:
                # If infiltration >= 1cm on previous day assume maximum soil evaporation
                r.EVS = EVSMX
                self._DSLR = 1
            else:
                # Else soil evaporation is a function days-since-last-rain (DSLR)
                EVSMXT = EVSMX * (sqrt(self._DSLR + 1) - sqrt(self._DSLR))
                r.EVS = min(EVSMX, EVSMXT + self._RINold)
                self._DSLR += 1

        # conductivities and Matric Flux Potentials for all layers
        pF = np.zeros_like(s.SM)
        conductivity = np.zeros_like(s.SM)
        
        matricfluxpot = np.zeros_like(s.SM)

        for i, layer in enumerate(self.soil_profile):
            pF[i] = layer.PFfromSM(s.SM[i])
            conductivity[i] = 10**layer.CONDfromPF(pF[i])
            matricfluxpot[i] = layer.MFPfromPF(pF[i])
            
        if self.soil_profile.GroundWater:
            # Equilibrium amounts
            EquilWater = np.zeros_like(s.SM)
            for i, layer in enumerate(self.soil_profile):
                SMsat = layer.SMfromPF(-1)
                WaterFromHeight = layer.WaterFromHeight
                LBSL = layer.DepthLowerBoundary
                TSL = layer.Thickness
                WC0 = layer.WC0                
                if(LBSL < s.ZT):
                    # Groundwater below layer
                    EquilWater[i] = WaterFromHeight(s.ZT-LBSL+TSL) - WaterFromHeight(s.ZT-LBSL)
                elif(LBSL-TSL < s.ZT):
                    # Groundwater in layer
                    EquilWater[i] = WaterFromHeight(s.ZT-LBSL+TSL) + (LBSL-s.ZT) * SMsat   
                else:
                    # Ground water above layer
                    EquilWater[i] = WC0
                # raise NotImplementedError("Groundwater influence not yet implemented.")

        # Potentially infiltrating rainfall
        if p.IFUNRN == 0:
            RINPRE = (1. - p.NOTINF) * drv.RAIN
        else:
            # infiltration is function of storm size (NINFTB)
            RINPRE = (1. - p.NOTINF * self.NINFTB(drv.RAIN)) * drv.RAIN


        # Second stage preliminary infiltration rate (RINPRE)
        # including surface storage and irrigation
        RINPRE = RINPRE + r.RIRR + s.SS
        if s.SS > 0.1:
            # with surface storage, infiltration limited by SOPE
            AVAIL = RINPRE + r.RIRR - r.EVW
            RINPRE = min(self.soil_profile.SurfaceConductivity, AVAIL)

        # maximum flow at Top Boundary of each layer
        # ------------------------------------------
        # DOWNWARD flows are calculated in two ways,
        # (1) a "dry flow" from the matric flux potentials
        # (2) a "wet flow" under the current layer conductivities and downward gravity.
        # Clearly, only the dry flow may be negative (=upward). The dry flow accounts for the large
        # gradient in potential under dry conditions (but neglects gravity). The wet flow takes into
        # account gravity only and will dominate under wet conditions. The maximum of the dry and wet
        # flow is taken as the downward flow, which is then further limited in order the prevent
        # (a) oversaturation and (b) water content to decrease below field capacity.
        #
        # UPWARD flow is just the dry flow when it is negative. In this case the flow is limited
        # to a certain fraction of what is required to get the layers at equal potential, taking
        # into account, however, the contribution of an upward flow from further down. Hence, in
        # case of upward flow from the groundwater, this upward flow is propagated upward if the
        # suction gradient is sufficiently large.

        FlowMX = np.zeros(len(s.SM) + 1)
        # first get flow through lower boundary of bottom layer
        if self.soil_profile.GroundWater:
            if(s.ZT >= self.soil_profile[-1].DepthLowerBoundary):
                
                # the old capillairy rise routine is used to estimate flow to/from the groundwater
                # note that this routine returns a positive value for capillairy rise and a negative
                # value for downward flow, which is the reverse from the convention in WATFDGW.                                
                from .subsol import calculate_flow_subsoil
                
                # Define some help variables
                CONTABLowestLayer = self.soil_profile[-1].CONDfromPF               
                pFLowestSoilLayer = self.soil_profile[-1].PFfromSM(s.SM[-1])
                depth_lowest_layer = self.soil_profile[-1].DepthLowerBoundary               
                depth_subsoil = self.soil_profile.SubSoilType.DepthLowerBoundary  
                HeightFromAir = self.soil_profile[-1].HeightFromAir                
                thickness_subsoil = self.soil_profile.SubSoilType.Thickness
                thickness_lowest_layer = self.soil_profile[-1].Thickness
                SubSM0 = self.soil_profile.SubSoilType.SM0
                WSUB0 = self.soil_profile.SubSoilType.WC0                   
                WaterFromHeight = self.soil_profile[-1].WaterFromHeight                
                WC0_lowest_layer = self.soil_profile[-1].WC0

                if(s.ZT > depth_lowest_layer):
                    # Groundwater below layered system
                    WSUB =  (depth_subsoil - s.ZT) * SubSM0 + WaterFromHeight(s.ZT - depth_lowest_layer)
                else:
                    # Saturated sub soil
                    WSUB = (depth_subsoil - depth_lowest_layer) * SubSM0
                    
                WC_lowest_layer = s.SM[-1] * thickness_lowest_layer
                
                D = s.ZT - depth_lowest_layer + thickness_lowest_layer/3.0
                SubFlow = calculate_flow_subsoil(pFLowestSoilLayer, D, CONTABLowestLayer)
                
                if(SubFlow >= 0):                    
                    # capillairy rise is limited by the amount required to reach equilibrium:
                    
                    # step 1. calculate equilibrium ZT for all air between ZT and top of laye
                    EqAir = WSUB0 - WSUB + (WC0_lowest_layer - WC_lowest_layer)                    
                    
                    # step 2. the groundwater level belonging to this amount of air in equilibrium
                    ZTeq1   = (depth_lowest_layer-thickness_lowest_layer) + HeightFromAir(EqAir)
                    
                    # step 3. this level should normally lie below the current level (otherwise there should
                    # not be capillairy rise). In rare cases however, due to the use of a mid-layer height
                    # in subroutine SUBSOL, a deviation could occur
                    ZTeq2   = max(s.ZT, ZTeq1)

                    # step 4. calculate for this ZTeq2 the equilibrium amount of water in the layer
                    WCequil = WaterFromHeight(ZTeq2-depth_lowest_layer+thickness_lowest_layer) - WaterFromHeight(ZTeq2-depth_lowest_layer)
                    
                    # step5. use this equilibrium amount to limit the upward flow
                    FlowMX[-1] = -1.0 * min (SubFlow, max(WCequil-WC_lowest_layer,0.0)/delt)
                else:
                    # downward flow ; air-filled pore space of subsoil limits downward flow
                    AirSub = (s.ZT-depth_lowest_layer)*SubSM0 - WaterFromHeight(s.ZT-depth_lowest_layer) 
                    FlowMX[-1] = min(abs(SubFlow), max(AirSub,0.0)/delt)
            else:
                FlowMX[-1] = 0.
        else:
            # Bottom layer conductivity limits the flow. Below field capacity there is no
            # downward flow, so downward flow through lower boundary can be guessed as
            FlowMX[-1] = max(self.soil_profile[-1].CondFC, conductivity[-1])

        # drainage
        DMAX = 0.0

        LIMDRY = np.zeros_like(s.SM)
        LIMWET = np.zeros_like(s.SM)
        TSL = [l.Thickness for l in self.soil_profile]
        for il in reversed(range(len(s.SM))):
            # limiting DOWNWARD flow rate
            # == wet conditions: the soil conductivity is larger
            #    the soil conductivity is the flow rate for gravity only
            #    this limit is DOWNWARD only
            # == dry conditions: the MFP gradient
            #    the MFP gradient is larger for dry conditions
            #    allows SOME upward flow
            if il == 0:  # Top soil layer
                LIMWET[il] = self.soil_profile.SurfaceConductivity
                LIMDRY[il] = 0.0
            else:
                # the limit under wet conditions is a unit gradient
                LIMWET[il] = (TSL[il-1]+TSL[il]) / (TSL[il-1]/conductivity[il-1] + TSL[il]/conductivity[il])

                # compute dry flow given gradients in matric flux potential
                if self.soil_profile[il-1] == self.soil_profile[il]:
                    # Layers il-1 and il have same properties: flow rates are estimated from
                    # the gradient in Matric Flux Potential
                    LIMDRY[il] = 2.0 * (matricfluxpot[il-1]-matricfluxpot[il]) / (TSL[il-1]+TSL[il])
                    if LIMDRY[il] < 0.0:
                        # upward flow rate ; amount required for equal water content is required below
                        MeanSM = (s.WC[il-1] + s.WC[il]) / (TSL[il-1]+TSL[il])
                        EqualPotAmount = s.WC[il-1] - TSL[il-1] * MeanSM  # should be negative like the flow
                else:
                    # iterative search to PF at layer boundary (by bisection)
                    il1  = il-1; il2 = il
                    PF1  = pF[il1]; PF2 = pF[il2]
                    MFP1 = matricfluxpot[il1]; MFP2 = matricfluxpot[il2]
                    for z in range(self.MaxFlowIter):  # Loop counter not used here
                        PFx = (PF1 + PF2) / 2.0
                        Flow1 = 2.0 * (+ MFP1 - self.soil_profile[il1].MFPfromPF(PFx)) / TSL[il1]
                        Flow2 = 2.0 * (- MFP2 + self.soil_profile[il2].MFPfromPF(PFx)) / TSL[il2]
                        if abs(Flow1-Flow2) < self.TinyFlow:
                            # sufficient accuracy
                            break
                        elif abs(Flow1) > abs(Flow2):
                            # flow in layer 1 is larger ; PFx must shift in the direction of PF1
                            PF2 = PFx
                        elif abs(Flow1) < abs(Flow2):
                            # flow in layer 2 is larger ; PFx must shift in the direction of PF2
                            PF1 = PFx
                    else:  # No break
                        msg = 'WATFDGW: LIMDRY flow iteration failed. Are your soil moisture and ' + \
                              'conductivity curves decreasing with increasing pF?'
                        raise exc.PCSEError(msg)
                    LIMDRY[il] = (Flow1 + Flow2) / 2.0

                    if LIMDRY[il] < 0.0:
                        # upward flow rate ; amount required for equal potential is required below
                        Eq1 = -s.WC[il2]; Eq2 = 0.0
                        for z in range(self.MaxFlowIter):
                            EqualPotAmount = (Eq1 + Eq2) / 2.0
                            SM1 = (s.WC[il1] - EqualPotAmount) / TSL[il1]
                            SM2 = (s.WC[il2] + EqualPotAmount) / TSL[il2]
                            PF1 = self.soil_profile[il1].SMfromPF(SM1)
                            PF2 = self.soil_profile[il2].SMfromPF(SM2)
                            if abs(Eq1-Eq2) < self.TinyFlow:
                                # sufficient accuracy
                                break
                            elif PF1 > PF2:
                                # suction in top layer 1 is larger ; absolute amount should be larger
                                Eq2 = EqualPotAmount
                            else:
                                # suction in bottom layer 1 is larger ; absolute amount should be reduced
                                Eq1 = EqualPotAmount
                        else:
                            msg = "WATFDGW: Limiting amount iteration in dry flow failed. Are your soil moisture " \
                                  "and conductivity curves decreasing with increase pF?"
                            raise exc.PCSEError(msg)

            FlowDown = True  # default
            if LIMDRY[il] < 0.0:
                # upward flow (negative !) is limited by fraction of amount required for equilibrium
                FlowMax = max(LIMDRY[il], EqualPotAmount * self.UpwardFlowLimit)
                if il > 0:
                    # upward flow is limited by amount required to bring target layer at equilibrium/field capacity
                    # if (il==2) write (*,*) '2: ',FlowMax, LIMDRY(il), EqualPotAmount * UpwardFlowLimit
                    if self.soil_profile.GroundWater:
                        # soil does not drain below equilibrium with groundwater
                        FCequil = max(self.soil_profile[il-1].WCFC, EquilWater[il-1])
                    else:
                        # free drainage
                        FCequil = self.soil_profile[il-1].WCFC

                    TargetLimit = WTRALY[il-1] + FCequil - s.WC[il-1]/delt
                    if TargetLimit > 0.0:
                        # target layer is "dry": below field capacity ; limit upward flow
                        FlowMax = max(FlowMax, -1.0 * TargetLimit)
                        # there is no saturation prevention since upward flow leads to a decrease of WC[il]
                        # instead flow is limited in order to prevent a negative water content
                        FlowMX[il] = max(FlowMax, FlowMX[il+1] + WTRALY[il] - s.WC[il]/delt)
                        FlowDown = False
                    elif self.soil_profile.GroundWater:
                        # target layer is "wet", above field capacity. Since gravity is neglected
                        # in the matrix potential model, this "wet" upward flow is neglected.
                        FlowMX[il] = 0.0
                        
                        # Implementation Allard states that FlowDown = True, but according to Rappoldt it is false. What is correct?
                        FlowDown = False
                    else:
                        # target layer is "wet", above field capacity, without groundwater
                        # The free drainage model implies that upward flow is rejected here.
                        # Downward flow is enabled and the free drainage model applies.
                        FlowDown = True

            if FlowDown:
                # maximum downward flow rate (LIMWET is always a positive number)
                FlowMax = max(LIMDRY[il], LIMWET[il])
                # this prevents saturation of layer il
                # maximum top boundary flow is bottom boundary flow plus saturation deficit plus sink
                FlowMX[il] = min(FlowMax, FlowMX[il+1] + (self.soil_profile[il].WC0 - s.WC[il])/delt + WTRALY[il])
        # end for

        r.RIN = min(RINPRE, FlowMX[0])

        # contribution of layers to soil evaporation in case of drought upward flow is allowed
        EVSL = np.zeros_like(s.SM)
        for il, layer in enumerate(self.soil_profile):
            if il == 0:
                EVSL[il] = min(r.EVS, (s.WC[il] - layer.WCW) / delt + r.RIN - WTRALY[il])
                EVrest = r.EVS - EVSL[il]
            else:
                Available = max(0.0, (s.WC[il] - layer.WCW)/delt - WTRALY[il])
                if Available >= EVrest:
                    EVSL[il] = EVrest
                    EVrest   = 0.0
                    break
                else:
                    EVSL[il] = Available
                    EVrest   = EVrest - Available
        # reduce evaporation if entire profile becomes airdry
        # there is no evaporative flow through lower boundary of layer NSL
        r.EVS = r.EVS - EVrest

        # Convert contribution of soil layers to EVS as an upward flux
        # evaporative flow (taken positive !!!!) at layer boundaries
        NSL = len(s.SM)
        EVflow = np.zeros_like(FlowMX)
        EVflow[0] = r.EVS
        for il in range(1, NSL):
            EVflow[il] = EVflow[il-1] - EVSL[il-1]
        EVflow[NSL] = 0.0  # see comment above

        # limit downward flows as to not get below field capacity / equilibrium content
        Flow = np.zeros_like(FlowMX)
        r.DWC = np.zeros_like(s.SM)
        Flow[0] = r.RIN - EVflow[0]
        for il, layer in enumerate(self.soil_profile):
            if self.soil_profile.GroundWater:
                # soil does not drain below equilibrium with groundwater
                WaterLeft = max(self.soil_profile[il].WCFC, EquilWater[il])
            else:
                # free drainage
                WaterLeft = layer.WCFC
            MXLOSS = (s.WC[il] - WaterLeft)/delt               # maximum loss
            Excess = max(0.0, MXLOSS + Flow[il] - WTRALY[il])  # excess of water (positive)
            Flow[il+1] = min(FlowMX[il+1], Excess - EVflow[il+1])  # note that a negative (upward) flow is not affected
            # rate of change
            r.DWC[il] = Flow[il] - Flow[il+1] - WTRALY[il]

        # Flow at the bottom of the profile
        r.BOTTOMFLOW = Flow[-1]

        # ! Percolation and Loss.
        # ! Equations were derived from the requirement that in the same layer, above and below
        # ! depth RD (or RDM), the water content is uniform. Note that transpiration above depth
        # ! RD (or RDM) may require a negative percolation (or loss) for keeping the layer uniform.
        # ! This is in fact a numerical dispersion. After reaching RDM, this negative (LOSS) can be
        # ! prevented by choosing a layer boundary at RDM.
        
        # Find index of deepest layers containing rooted soil (ILR) and deepest layer that will contain roots (ILM)
        # ILR = len(self.soil_profile) - 1
        # ILM = len(self.soil_profile) - 1
        # RD = self._determine_rooting_depth()
        # for il, layer in reversed(list(enumerate(self.soil_profile))):
        #     if(layer.DepthLowerBoundary >= RD):
        #         ILR = il
        #     if(layer.DepthLowerBoundary >= self._RDM):
        #         ILM = il
        
        # if(ILR < ILM):
        #     # layer ILR is devided into rooted part (where the sink is) and a below-roots part
        #     # The flow in between is PERC            
        #     f1 = self.soil_profile[ILR].Wtop
        #     PERC = (1.0-f1) * (Flow[ILR]-WTRALY[ILR]) + f1 * Flow[ILR+1]
            
        #     # layer ILM is divided as well ; the flow in between is LOSS
        #     f1 = self.soil_profile[ILM].Wpot
        #     LOSS = (1.0-f1) * Flow[ILM] + f1 * Flow[ILM+1]
        
        # elif(ILR == ILM):
        #     f1 = self.soil_profile(ILR).Wtop
        #     f2 = self.soil_profile(ILM).Wpot
        #     f3 = 1.0 - f1 - f2
        #     LOSS = f3 * (Flow[ILR]-WTRALY[ILR]) + (1.0-f3) * Flow[ILR+1]
        #     PERC = (1.0-f1) * (Flow[ILR]-WTRALY[ILR]) + f1 * Flow[ILR+1]
        # else:
        #     msg = "The index of the lowest rootable soil layer cannot be smaller than the index of the lowest rooted soil layer"
        #     raise Exception(msg)
            
        # r.PERC = PERC

        if self.soil_profile.GroundWater:
            # groundwater influence
            # DWBOT = LOSS - Flow[-1]
            DWSUB = Flow[-1]
            r.DWSUB = DWSUB
            pass

        # Computation of rate of change in surface storage and surface runoff
        # SStmp is the layer of water that cannot infiltrate and that can potentially
        # be stored on the surface. Here we assume that RAIN_NOTINF automatically
        # ends up in the surface storage (and finally runoff).
        SStmp = drv.RAIN + r.RIRR - r.EVW - r.RIN
        # rate of change in surface storage is limited by SSMAX - SS
        r.DSS = min(SStmp, (p.SSMAX - s.SS))
        # Remaining part of SStmp is send to surface runoff
        r.DTSR = SStmp - r.DSS
        # incoming rainfall rate
        r.DRAINT = drv.RAIN

        self._RINold = r.RIN
        r.Flow = Flow

    @prepare_states
    def integrate(self, day, delt):
        p = self.params
        s = self.states
        k = self.kiosk
        r = self.rates

        # amount of water in soil layers ; soil moisture content
        SM = np.zeros_like(s.SM)
        WC = np.zeros_like(s.WC)
        for il, layer in enumerate(self.soil_profile):
            WC[il] = s.WC[il] + r.DWC[il] * delt
            SM[il] = WC[il] / layer.Thickness
        # NOTE: We cannot replace WC[il] with s.WC[il] above because the kiosk will not
        # be updated since traitlets cannot monitor changes within lists/arrays.
        # So we have to assign:
        s.SM = SM
        s.WC = WC

        # total transpiration
        s.WTRAT += r.WTRA * delt

        # total evaporation from surface water layer and/or soil
        s.EVWT += r.EVW * delt
        s.EVST += r.EVS * delt

        # totals for rainfall, irrigation and infiltration
        s.RAINT += self._RAIN
        s.TOTINF += r.RIN * delt
        s.TOTIRR += r.RIRR * delt

        # surface storage and runoff
        s.SS += r.DSS * delt
        s.TSR += r.DTSR * delt

        # loss of water by outflow through bottom of profile
        s.BOTTOMFLOWT += r.BOTTOMFLOW * delt

        # percolation from rootzone ; interpretation depends on mode
        # if self.soil_profile.GroundWater:
        #     # with groundwater this flow is either percolation or capillary rise
        #     if r.PERC > 0.0:
        #         s.PERCT = s.PERCT + r.PERC * delt
        #     else:
        #         s.CRT = s.CRT - r.PERC * delt
        # else:
        #     # without groundwater this flow is always called percolation
        #     s.CRT = 0.0

        # change of rootzone
        RD = self._determine_rooting_depth()
        if abs(RD - self._RDold) > 0.001:
            self.soil_profile.determine_rooting_status(RD, self._RDM)

        # compute summary values for rooted, potentially rooted and unrooted soil compartments
        W = 0.0 ; WAVUPP = 0.0
        WLOW = 0.0 ; WAVLOW = 0.0
        WBOT = 0.0 ; WAVBOT = 0.0
        # get W and WLOW and available water amounts
        for il, layer in enumerate(self.soil_profile):
            W += s.WC[il] * layer.Wtop
            WLOW += s.WC[il] * layer.Wpot
            WBOT += s.WC[il] * layer.Wund
            WAVUPP += (s.WC[il] - layer.WCW) * layer.Wtop
            WAVLOW += (s.WC[il] - layer.WCW) * layer.Wpot
            WAVBOT += (s.WC[il] - layer.WCW) * layer.Wund
            
        if(self.soil_profile.GroundWater):
            WSUB = s.WSUB
            WSUB+= r.DWSUB * delt

        
        # Update states
        s.W = W
        s.WLOW = WLOW
        s.WWLOW = s.W + s.WLOW
        s.WBOT = WBOT
        s.WAVUPP = WAVUPP
        s.WAVLOW = WAVLOW
        s.WAVBOT = WAVBOT
        
        if(self.soil_profile.GroundWater):
            # Update ground water amount
            s.WSUB = WSUB
            
            # Update ground water level
            ZTfound = False
            for il, layer in reversed(list(enumerate(self.soil_profile))):
                HeightFromAir = layer.HeightFromAir
                if(il == len(self.soil_profile) - 1):                    
                    WSUB0 = self.soil_profile.SubSoilType.WC0
                    AirSub = WSUB0 - s.WSUB
                    if(AirSub > 0.01):
                        # groundwater is in subsoil which is not completely saturated
                        s.ZT = min(layer.DepthLowerBoundary + HeightFromAir(AirSub), self.soil_profile.SubSoilType.DepthLowerBoundary)
                        ZTfound = True
                        break
                    if(s.SM[il] < 0.99 * layer.SM0):
                         # groundwater is in this layer
                         s.ZT = layer.DepthLowerBoundary - layer.Thickness + min(layer.Thickness,  HeightFromAir(layer.WC0-s.WC[il]))
                         ZTfound = True
                         break
            if(ZTfound == False):
                # entire system saturated
                s.ZT = 0.0
        # save rooting depth for which layer contents have been determined
        self._RDold = RD
        s.SM_MEAN = s.W/RD

    @prepare_states
    def finalize(self, day):
        s = self.states
        p = self.params
        checksum = (p.SSI - s.SS  # change in surface storage
                    + self._WCI - s.WC.sum()  # Change in soil water content
                    + s.RAINT + s.TOTIRR  # inflows to the system
                    - s.WTRAT - s.EVWT - s.EVST - s.TSR - s.BOTTOMFLOWT  # outflows from the system
                    )
        if abs(checksum) > 0.0001:
            msg = "Waterbalance not closing on %s with checksum: %f" % (day, checksum)
            raise exc.WaterBalanceError(msg)

    def _determine_rooting_depth(self):
        """Determines appropriate use of the rooting depth (RD)

        This function includes the logic to determine the depth of the upper (rooted)
        layer of the water balance. See the comment in the code for a detailed description.
        """
        if "RD" in self.kiosk:
            return self.kiosk["RD"]
        else:
            # Hold RD at default value
            return self._default_RD

    def _on_CROP_START(self):
        self.crop_start = True

    def _on_CROP_FINISH(self):
        pass
        # self.in_crop_cycle = False
        # self.rooted_layer_needs_reset = True

    def _on_IRRIGATE(self, amount, efficiency):
        self._RIRR = amount * efficiency

    def _setup_new_crop(self):
        """Retrieves the crop maximum rootable depth, validates it and updates the rooting status
        in order to have a correct calculation of the summary waterbalance states.

        """
        self._RDM = self.parameter_provider["RDMCR"]
        self.soil_profile.validate_max_rooting_depth(self._RDM)
        self.soil_profile.determine_rooting_status(self._default_RD, self._RDM)