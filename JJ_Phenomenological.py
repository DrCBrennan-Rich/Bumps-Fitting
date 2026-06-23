# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (6) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=L11_NathanIntermediate3 --session=JJSession.h5 JJ_Phenomenologica.py

import bumps.names as bmp
import numpy as np
from scipy.integrate import quad

#Coherence lengths
xi_F1=0.3 #nm
xi_F2=0.16 #nm

def line(x, Gradient, Intercept):
    return Gradient*x + Intercept

def JC_model(d_F, Amplitude, xi_F1, xi_F2, d_0pi):
    
    SinTerm = np.sin((d_F-d_0pi)/xi_F2)
    
    return Amplitude*(np.exp(-d_F/xi_F1)*np.abs(SinTerm))



#Load the data from the file Data.txt
d_F,y,dy = np.loadtxt('L11 data 4.2K.txt').T 


Model = bmp.Curve(
    JC_model,
    d_F, y, dy,
    Amplitude=10000, 
    xi_F1=0.3, 
    xi_F2=0.16, 
    d_0pi=1.0,
)


### Limits of fitting values ###

Model.Amplitude.range(90,350)
Model.d_0pi.range(0.0,0.99*np.pi*xi_F2)

Model.xi_F1.range(0.1,0.3)
Model.xi_F2.range(0.1,0.2)

#Model.xi_F1.dev(std=0.1, mean=0.3, limits=None)
#Model.xi_F2.dev(std=0.1, mean=0.16, limits=None)

#######

#Initial values

Model.Amplitude.value = 130
Model.d_0pi.value = 1

Model.xi_F1.value = 0.1
Model.xi_F2.value = 0.16


problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()


import matplotlib.pyplot as plt

for d_0pi_test in [0.2,0.5,1,2,3]:
    ytest = JC_model(
        d_F,
        Amplitude=10000, 
        xi_F1=1.0, 
        xi_F2=1.0, 
        d_0pi=d_0pi_test, 
    )
    plt.plot(d_F, ytest, label=f"d_0pi={d_0pi_test}")

plt.legend()
plt.show()
