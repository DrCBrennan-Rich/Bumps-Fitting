# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (6) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=ExportFolder --session=JJSession.h5 JJ_Phenomenological.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

JunctionResistance = 1.55E-3 #Ohms
#Fitting Paramaters
Amplitude = 227.201 #Current amplitude
CoherenceLength_F1=0.419556 #nm
CoherenceLength_F2=0.15877 #nm
d_0pi = 0.277311 #phase

def JC_model(d_F, Amplitude, CoherenceLength_F1, CoherenceLength_F2, d_0pi):
    """Calculate the critical voltage across the Josephson junction according
    to a phenomenological model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq. 6 of 
    the paper by Birge and Satchell: https://doi.org/10.1063/5.0195229.
    It models a sinosoidal oscillation with a superimposed exponential
    decay. 

    Args:
        d_F (numpy.ndarray): List of (float) thicknesses of the ferromagnetic 
            junction (nm).
        Amplitude (float): Amplitude of the fitted function (uV).
        CriticalTemperature (float): Critical temperature of the 
            superconductor (K).
        CoherenceLength (float): Coherence length in the ferromagnet (nm).
        DeadLayers (float): Thickness of dead (non-magnetic) material in the ferromagnet. 
            Negative values indicate increased effective ferromagnetic thickness 
            due to proximity magnetisation in the normal metal (nm). 
        Amplitude (float): If provided, can be used to set an arbitrary scaled
            amplitude for the output.

    Returns:
        IcRn (float): Voltage across the Josephson junction (uV).

    Notes:
        Equation being solved is IcRn = pi*SC_gap^2*F(x)/(4*T_c)
        where: F(x) = 2x*(Cos(x)Sinh(x) + Sin(x)Cosh(x))/(Cosh(2x) - Cos(2x))
    """
    SinTerm = np.sin((d_F-d_0pi)/CoherenceLength_F2)
    
    return Amplitude*(np.exp(-d_F/CoherenceLength_F1)*np.abs(SinTerm))

#Load the data from the file Data.txt
#d_F,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T

OrderingIndex = np.argsort(d_F)
d_F = d_F[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

#y = y/JunctionResistance
#dy = dy/JunctionResistance

Model = bmp.Curve(
    JC_model,
    d_F, y, dy,
    Amplitude=100, 
    CoherenceLength_F1=0.3, 
    CoherenceLength_F2=0.16, 
    d_0pi=1.0)

### Limits of fitting values ###

Model.Amplitude.range(1,1000)
Model.d_0pi.range(0.25,0.29)#(0.0,0.9*np.pi*CoherenceLength_F2) #Due to the periodicity of d_0pi, this will be the paramter range
Model.CoherenceLength_F1.range(0.3,0.45)
Model.CoherenceLength_F2.range(0.1,0.2)

#Model.CoherenceLength_F1.dev(std=0.1, mean=0.3, limits=None)
#Model.CoherenceLength_F2.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values

Model.Amplitude.value = 100
Model.d_0pi.value = 0.332473
Model.CoherenceLength_F1.value = 0.599986
Model.CoherenceLength_F2.value = 0.146829

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot

plt.errorbar(
    d_F, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.1, 4, 1000)

for d_0pi_test in [0.361262]:
    ytest = JC_model(
        X_axis,
        Amplitude= Amplitude, 
        CoherenceLength_F1= CoherenceLength_F1,
        CoherenceLength_F2=CoherenceLength_F2, 
        d_0pi=d_0pi, 
    )
    plt.plot(X_axis, ytest, label=f"Fitted Curve", linewidth=3)
    
plt.yscale("log")
plt.xlabel("Thickness (nm)")
plt.ylabel("Current (mA)")
plt.legend()
plt.show()
