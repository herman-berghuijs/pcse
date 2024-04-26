from math import log10, exp

def calculate_flow_subsoil(PF,D,CONTAB):
    # ----------------------------------------------------------------------- #
# Original description: 
# ----------------------------------------------------------------------- #
# $Id: subsol.for 1.2 1997/10/02 15:20:35 LEM release $

#     Chapter 15 in documentation WOFOST Version 4.1 (1988)

#     This routine calculates the rate of capillary flow or
#     percolation between groundwater table and root zone. The
#     stationary flow is found by integration of
#            dZ = K.d(MH)/(K + FLW)   ,where
#     Z= height above groundwater, MH= matric head, K= conductivity and
#     FLW= chosen flow. In an iteration loop the correct flow is found.
#     The integration goes at most over four intervals : [0,45],[45,170],
#     [170,330] and [330, MH-rootzone] (last one on logarithmic scale).

#     Subroutines and functions called: AFGEN.
#     Called by routine WATGW.

#     Author: C. Rappoldt, January 1986, revised June 1990  
# ---------------------------------------------------------------------- #
# Python translation
#
#     Author: H.N.C. Berghuijs, April 2024
# ----------------------------------------------------------------------- #
    # 15.1 Declarations and constants  
    ELOG10 = 2.302585
    PGAU = [0.1127016654, 0.5, 0.8872983346]
    WGAU = [0.2777778, 0.4444444, 0.2777778]
    START = [0., 45., 170., 330.]
    LOGST4 = 2.518514
    PFSTAN = [0.705143, 1.352183, 1.601282, 1.771497, 2.031409, 
                2.192880,2.274233,2.397940,2.494110]
    
    DEL = [0] * 4
    CONDUC = [0] * 12
    PFGAU = [0] * 12
    HULP = [0] * 12

    # 15.2 Calculation of matric head and check on small pF
    PF1 = PF
    D1 = D
    MH = exp(ELOG10 * PF1)
    
    if(PF1 > 0.):
        IINT = 0
        #15.3 number and width of integration intervals
        for I1 in range(1, 4+1):
            if(I1<=3):
                DEL[I1-1] = min(START[I1+1-1],MH)-START[I1-1]
            if(I1==4):
                DEL[I1-1] = PF1 - LOGST4
            if(DEL[I1-1] <= 0.):
                break
            IINT+=1
        
        #15.4 preparation of three-point Gaussian integration
        for I1 in range(1, IINT+1):
            for I2 in range(1, 3+1):
                I3 = 3 * (I1 - 1) + I2
                if(I1 == IINT):
                    if(IINT <= 3):
                        PFGAU[I3-1] = log10(START[IINT-1] + PGAU[I2-1] * DEL[IINT - 1])
                    if(IINT == 4):
                        PFGAU[I3-1] = LOGST4 + PGAU[I2-1] * DEL[IINT-1]
                    # variables needed in the loop below    
                    CONDUC[I3-1] = exp(ELOG10*CONTAB(PFGAU[I3-1]))
                    HULP[I3-1]   = DEL[I1-1]*WGAU[I2-1]*CONDUC[I3-1]
                    if(I3>9):
                        HULP[I3-1] = HULP[I3-1]*ELOG10*exp(ELOG10*PFGAU[I3-1]) 
                else:
                    # The three points in the full-width intervals are standard
                    PFGAU[I3-1] = PFSTAN[I3-1]
                    # variables needed in the loop below
                    CONDUC[I3-1] = exp(ELOG10*CONTAB(PFGAU[I3-1]))
                    HULP[I3-1]   = DEL[I1-1]*WGAU[I2-1]*CONDUC[I3-1]
                    if(I3>9):
                        HULP[I3-1] = HULP[I3-1]*ELOG10*exp(ELOG10*PFGAU[I3-1])
        
        # 15.5 setting upper and lower limit
        FU = 1.27
        FL = -1.0 * exp(ELOG10 * CONTAB(PF1))
        if(MH<=D1):
            FU = 0.
        if(MH >= D1):
            FL = 0.
        if(MH == D1):
            FLOW = (FU + FL) / 2.
        else:
            # 15.6 Iteration loop
            IMAX = 3 * IINT
            for i1 in range(1, 15+1):
                FLW = (FU + FL) / 2.
                DF = (FU - FL) / 2.
                if((DF < 0.01) & ((DF/abs(FLW)) < 0.1)):
                    FLOW = (FU + FL) / 2.
                else:
                    Z = 0.
                    for I2 in range(1, IMAX + 1):
                        Z = Z + HULP[I2 - 1] / (CONDUC[I2-1] + FLW)
                    if(Z >= D1):
                        FL = FLW
                    if(Z <= D1):
                        FU = FLW
                    FLOW = (FU + FL) / 2.
    else:
        # In case of small matric head
        K0 = exp(ELOG10 * CONTAB(-1.))
        FLOW = K0*(MH/D-1.)        
    return FLOW
    
