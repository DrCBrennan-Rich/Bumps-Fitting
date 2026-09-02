# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (3 & 4) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=ExportFolder --session=JJSession.h5 JJ_Kuprianov.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
plt.rcParams.update({'font.size': 40})

k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K
JunctionResistance = 1.55E-3 #Ohms

#Coherence lengths
#Amplitude = 260.452 #Current amplitude
CriticalTemperature = 8.87267 #K
CoherenceLength = 0.36178 #nm
SC_gap =  0.00159515 #eV
Amplitude = 2.36283e-05

def JC_Dirty_Limit(d_F, SC_gap, CriticalTemperature, CoherenceLength,
                   Amplitude=Amplitude):
    """Calculate the critical voltage across the Josephson junction according
    to a diffusive model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq. 3 of 
    the paper by Birge and Satchell: https://doi.org/10.1063/5.0195229.
    It is derived from the Usadel equations and is therefore valid in the 
    regime where impurity scattering lengths are significantly shorter than
    normal, superconductor, or ferromagnetic coherence lengths or dimensions of
    the system.

    Args:
        d_F (numpy.ndarray): List of (float) thicknesses of the ferromagnetic 
            junction (nm).
        SC_gap (float): Superconducting gap (eV).
        CriticalTemperature (float): Critical temperature of the 
            superconductor (K).
        CoherenceLength (float): Coherence length in the ferromagnet (nm).
        Amplitude (float): If provided, can be used to set an arbitrary scaled
            amplitude for the output.

    Returns:
        IcRn (float): Voltage across the Josephson junction (uV).

    Notes:
        Equation being solved is IcRn = pi*SC_gap^2*F(x)/(4*T_c)
        where: F(x) = 2x*(Cos(x)Sinh(x) + Sin(x)Cosh(x))/(Cosh(2x) - Cos(2x))
    """
    
    x = d_F/CoherenceLength
    
    Numerator = np.cos(x)*np.sinh(x) + np.sin(x)*np.cosh(x)
    
    Denominator = np.cosh(2*x) - np.cos(2*x)
    
    if Amplitude is None:
        Amplitude = 2*np.pi*SC_gap*SC_gap/(4*k_B*CriticalTemperature)
    
    IcRn = Amplitude*x*np.abs(Numerator/Denominator)
    
    return IcRn*1E6 #IcRn in uV

#In the limit of large x and Temperature ~ CriticalTemperature
def JC_Dirty_Limit_Simplified(d_F, SC_gap, CriticalTemperature,
                              CoherenceLength, Amplitude=Amplitude):
    """Calculate the critical voltage across the Josephson junction according
    to a simplifed diffusive model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq. 4 of 
    the paper by Birge and Satchell: https://doi.org/10.1063/5.0195229.
    It is derived from the Usadel equations and is therefore valid in the 
    regime where impurity scattering lengths are significantly shorter than
    normal, superconductor, or ferromagnetic coherence lengths or dimensions of
    the system. It also has the extra requirement that the ferromagnetic 
    thickness is significantly larger than the feromagnetic coherence length 
    and for temperatures near the critical temperature of the superconductor.

    Args:
        d_F (numpy.ndarray): List of (float) thicknesses of the ferromagnetic 
            junction (nm).
        SC_gap (float): Superconducting gap (eV).
        CriticalTemperature (float): Critical temperature of the 
            superconductor (K).
        CoherenceLength (float): Coherence length in the ferromagnet (nm).
        Amplitude (float): If provided, can be used to set an arbitrary scaled
            amplitude for the output.

    Returns:
        IcRn (float): Voltage across the Josephson junction (uV).

    Notes:
        Equation being solved is IcRn = pi*SC_gap^2*Sinc[d_F/CoherenceLength]/(4*T)
    """
    
    x = d_F/CoherenceLength
    
    SinTerm = np.sin(x + np.pi/4)
    
    if Amplitude is None:
        Amplitude = np.sqrt(2)*SC_gap*SC_gap/(4*k_B*CriticalTemperature)
    
    IcRn = Amplitude*x*np.exp(-x)*np.abs(SinTerm)
    
    return IcRn*1E6 #IcRn in uV

#Load the data from the file Data.txt
#d_F,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T    

OrderingIndex = np.argsort(d_F)
d_F = d_F[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

#y = y/JunctionResistance
#dy = dy/JunctionResistance

Model = bmp.Curve(
    JC_Dirty_Limit_Simplified,
    d_F, y, dy,
    SC_gap = SC_gap, 
    CriticalTemperature = CriticalTemperature, 
    CoherenceLength = CoherenceLength,
    #Amplitude = Amplitude
    )

### Limits of fitting values ###

Model.SC_gap.range(1.4E-3,1.6E-3)
Model.CriticalTemperature.range(8.5,9.5)
Model.CoherenceLength.range(0.001,5)

Model.Amplitude.range(1E-6,1E-3)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)

#######
#Initial values

Model.CriticalTemperature.value = CriticalTemperature
Model.SC_gap.value = SC_gap
Model.CoherenceLength.value = CoherenceLength

Model.Amplitude.value = Amplitude

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot

plt.errorbar(
    d_F, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.25, 2.4, 1000)

for CoherenceLength_test in [0.361262]:
    ytest = JC_Dirty_Limit_Simplified(
        X_axis,
        SC_gap = SC_gap, 
        CriticalTemperature = CriticalTemperature, 
        CoherenceLength = CoherenceLength,
        Amplitude=Amplitude)
    
    plt.plot(X_axis, ytest, label=f"Fitted Curve Simplifed", linewidth=3)
    
plt.yscale("log")
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel(r"$I_cR_N$ ($\mathrm{\mu V}$)", fontsize=34)
plt.legend()
plt.show()
