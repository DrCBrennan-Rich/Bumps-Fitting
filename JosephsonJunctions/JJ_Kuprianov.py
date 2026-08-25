# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (6) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=ExportFolder --session=JJSession.h5 JJ_Kuprianov.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
plt.rcParams.update({'font.size': 40})

k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K
JunctionResistance = 1.55E-3 #Ohms

#Coherence lengths
Amplitude = 260.452 #Current amplitude
CriticalTemperature = 9.2 #K
CoherenceLength= 0.332117 #nm
SC_gap = 1.5E-3 #eV

def JC_Dirty_Limit(d_F, SC_gap, CriticalTemperature, CoherenceLength):
    
    x = d_F/CoherenceLength
    
    Numerator = np.cos(x)*np.sinh(x) + np.sin(x)*np.cosh(x)
    
    Denominator = np.cosh(2*x) - np.cos(2*x)
    
    Amplitude = 2*np.pi*SC_gap*SC_gap/(4*k_B*CriticalTemperature)
    
    IcRn = Amplitude*x*np.abs(Numerator/Denominator)
    
    return IcRn

#In the limit of large x and Temperature ~ CriticalTemperature
def JC_Dirty_Limit_Simplified(d_F, SC_gap, CriticalTemperature, CoherenceLength):
    
    x = d_F/CoherenceLength
    
    SinTerm = np.sin(x + np.pi/4)
    
    Amplitude = np.sqrt(2)*SC_gap*SC_gap/(4*k_B*CriticalTemperature)
    
    IcRn = Amplitude*x*np.exp(-x)*np.abs(SinTerm)
    
    return IcRn

#Load the data from the file Data.txt
#d_F,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T

OrderingIndex = np.argsort(d_F)
d_F = d_F[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

#y = y/JunctionResistance
#dy = dy/JunctionResistance

Model = bmp.Curve(
    JC_Dirty_Limit,
    d_F, y, dy,
    SC_gap = SC_gap, 
    CriticalTemperature = CriticalTemperature, 
    CoherenceLength = CoherenceLength)

### Limits of fitting values ###

Model.SC_gap.range(1.4E-3,1.6E-3)
Model.CriticalTemperature.range(9.0,9.3)
Model.CoherenceLength.range(0.1,1.5)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)

#######
#Initial values

Model.CriticalTemperature.value = CriticalTemperature
Model.SC_gap.value = SC_gap
Model.CoherenceLength.value = CoherenceLength


problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot

plt.errorbar(
    d_F, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.1, 2.5, 1000)

for CoherenceLength_test in [0.361262]:
    ytest = JC_Dirty_Limit(
        X_axis,
        SC_gap = SC_gap, 
        CriticalTemperature = CriticalTemperature, 
        CoherenceLength = CoherenceLength)
    
    plt.plot(X_axis, ytest, label=f"Fitted Curve Simplifed", linewidth=3)
    
plt.yscale("log")
plt.xlabel("Thickness (nm)")
plt.ylabel("Current (mA)")
plt.legend()
plt.show()
