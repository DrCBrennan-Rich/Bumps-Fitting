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

JunctionResistance = 1.55E-3 #Ohms

#Coherence lengths
CoherenceLength_F1=0.3 #nm
CoherenceLength_F2=0.16 #nm

def JC_model(d_F, Amplitude, CoherenceLength_F1, CoherenceLength_F2, d_0pi):
    
    SinTerm = np.sin((d_F-d_0pi)/CoherenceLength_F2)
    
    return Amplitude*(np.exp(-d_F/CoherenceLength_F1)*np.abs(SinTerm))

#Load the data from the file Data.txt
#d_F,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T

d_F = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 1.3, 2.45, 2.9, 3.05, 0.25, 0.3001, 0.35, 0.65, 0.75, 0.85, 3.35, 3.5, 3.95, 4.1])
y = np.array([65.340875, 24.878000000000004, 51.64000000000001, 61.19333333333333, 39.726, 25.309341886259293, 14.174717467474276, 18.957235235409815, 13.972562686109798, 10.008982510530531, 6.939496641380081, 2.3033272230925714, 6.3283471086318865, 1.3283048839258265, 2.1937826740803543, 2.6960435520171293, 2.642885048481178, 26.75475, 18.771833333333337, 48.66025, 45.33733333333334, 11.033598167685819, 19.0885, 2.151310036922145, 5.327660826872811, 2.3257632595999977, 2.2332393553653582])
dy = np.array([3.906590843129723, 0.7540000000000013, 2.8450014645573267, 1.514830390212418, 0.9546742245394503, 1.624217115530211, 0.9081232687658485, 0.3261137420414092, 0.779596233884249, 0.8897751989591207, 0.23563425164791063, 0.28325155792176143, 0.4182558888900494, 0.26331238028429693, 0.21384513210250294, 0.3840692332507965, 0.204401721667579, 0.7852499999999994, 2.3378397495218644, 5.166759346614983, 5.5504418943199685, 0.3490574302442736, 1.2972500000000018, 0.09114521005284514, 0.5093084657670612, 0.11055011898099443, 0.3681883271793756])

OrderingIndex = np.argsort(d_F)
d_F = d_F[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

y = y/JunctionResistance
dy = dy/JunctionResistance

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
