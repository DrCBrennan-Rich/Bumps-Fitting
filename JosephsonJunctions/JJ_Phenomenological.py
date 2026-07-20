# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (6) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=ExportFolder --session=JJSession.h5 JJ_Phenomenological.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})

#Coherence lengths
CoherenceLength_F1=0.3 #nm
CoherenceLength_F2=0.16 #nm

def JC_model(d_F, Amplitude, CoherenceLength_F1, CoherenceLength_F2, d_0pi):
    
    SinTerm = np.sin((d_F-d_0pi)/CoherenceLength_F2)
    
    return Amplitude*(np.exp(-d_F/CoherenceLength_F1)*np.abs(SinTerm))

#Load the data from the file Data.txt
d_F,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T

OrderingIndex = np.argsort(d_F)
d_F = d_F[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

y = y/1.9E-3
dy = dy/1.9E-3

Model = bmp.Curve(
    JC_model,
    d_F, y, dy,
    Amplitude=10000, 
    CoherenceLength_F1=0.3, 
    CoherenceLength_F2=0.16, 
    d_0pi=1.0)

### Limits of fitting values ###

Model.Amplitude.range(20/1.9E-3,120/1.9E-3)
Model.d_0pi.range(0.0,0.9*np.pi*CoherenceLength_F2) #Due to the periodicity of d_0pi, this will be the paramter range
Model.CoherenceLength_F1.range(0.1,0.6)
Model.CoherenceLength_F2.range(0.1,0.2)

#Model.CoherenceLength_F1.dev(std=0.1, mean=0.3, limits=None)
#Model.CoherenceLength_F2.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values

Model.Amplitude.value = 120000
Model.d_0pi.value = 0.35
Model.CoherenceLength_F1.value = 0.4
Model.CoherenceLength_F2.value = 0.17

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot

plt.errorbar(
    d_F, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.1, 2, 1000)

for d_0pi_test in [0.361262]:
    ytest = JC_model(
        X_axis,
        Amplitude= 23979.9, 
        CoherenceLength_F1= 0.311356,
        CoherenceLength_F2=0.163457, 
        d_0pi=d_0pi_test, 
    )
    plt.plot(X_axis, ytest, label=f"Fitted Curve", linewidth=3)
    
plt.yscale("log")
plt.xlabel("Thickness (nm)")
plt.ylabel("Current (mA)")
plt.legend()
plt.show()
