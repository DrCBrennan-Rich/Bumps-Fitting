# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (1) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_Ballistic.py

#### WARNING ####

#The function JC_model has been written to be fast and easily callable thousands
#of times. This is at the cost of stability if the ferromagnetic coherence
#length is smaller than 0.2 nm (see associated Figure on GitHub):
#https://github.com/DrCBrennan-Rich/Bumps-Fitting/blob/main/JosephsonJunctions/Ballistic_CoherenceLengthComparison.pdf
#And so this should be set as the lower fitting bound.
#In situations where the coherence length is approaching this limit, it would be advisable to use:
#https://github.com/DrCBrennan-Rich/Bumps-Fitting/blob/main/JosephsonJunctions/JJ_Ballistic_simplified.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid


T = 5 #Temperature in K
T_c = 10 #Critical temperature in K
alpha = 1
y_max = 10*alpha

Resistance = 1.4E-3 #ohms

k_B = 1.38E-23
T = 5 #K
e = 1.6E-19

Beta = 1/(k_B*T)

def JC_model(Thickness, CoherenceLength, SC_gap):
    
    PhiList = np.linspace(0,2*np.pi,60,endpoint=False)[:,None]
    
    AlphaList = np.maximum(Thickness/CoherenceLength, 1e-6)
    
    Ic = np.zeros_like(AlphaList)
    
    for i,Alpha in enumerate (AlphaList):
        
        yMax = max(10*Alpha,5)
        Step = max(200,int(100*yMax))
        yList = np.linspace(Alpha,yMax,Step)[None,:]
        
        SinMinus = np.sin(0.5*(PhiList-yList))
        SinPlus = np.sin(0.5*(PhiList+yList))
        
        TanhMinus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList-yList)))
        TanhPlus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList+yList)))
        
        Integrand = (1/(yList*yList*yList))*(SinMinus*TanhMinus+SinPlus*TanhPlus)
        
        Iphi = (np.pi*SC_gap*Alpha*Alpha/(2*Resistance)*trapezoid(Integrand, x=yList, axis=1))
        
        Ic[i] = np.max(np.abs(Iphi))
        
    return Ic
        

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('L11 data 4.2K.txt').T
#y = np.log(y)
Model = bmp.Curve(JC_model, d, y, dy)

### Limits of fitting values ###

Model.SC_gap.range(1,50) 
Model.CoherenceLength.range(0.001,100)  

#Model.Gradient.dev(std=0.05, mean=None, limits=None)
#Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values

Model.SC_gap.value = 31 #eV
Model.CoherenceLength.value = 0.2

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Plot data points if desired
'''
plt.errorbar(
    d, y, yerr=dy,
    fmt='o',
    capsize=3,
    label='Experimental data'
)
'''
x_axis = np.linspace(0.01,10,100)

for CoherenceLength_test in [8E-2, 20E-2, 5E-1, 1]:
    ytest = JC_model(
        x_axis,
        CoherenceLength=CoherenceLength_test,
        SC_gap = 1.5E-3
    )
    plt.plot(x_axis, ytest, label=f"Coherence Length={CoherenceLength_test}")
    plt.yscale('log')

plt.xlabel("Ferromagnet thickness (nm)")
plt.ylabel(r"Critical current density $J_c$ (mA/m$^2$)")

plt.legend()
plt.savefig("CoherenceLengthComparison.svg", format="svg")
plt.show()
'''
a = 10.5E-2
for SC_gap_test in [1.5E-3,4E-3,8E-3,12E-3]:
    ytest = JC_model(
        d,
        CoherenceLength=a,
        SC_gap = SC_gap_test
    )
    plt.plot(d, ytest, label=f"Gap={SC_gap_test} and {a}")

plt.legend()
plt.show()
'''
