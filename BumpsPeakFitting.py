# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 01:50:56 2025
@author: pycbr
"""
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=PeakOutput --session=PeakSession.h5 BumpsPeakFitting.py

import bumps.names as bmp
import numpy as np

def Gaussian(x, Mu, Sigma, Amplitude):
    
    return Amplitude*(1/(Sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-Mu)/Sigma)**2)

x,y,dy = np.loadtxt('PeakData.txt').T
Model = bmp.Curve(Gaussian, x, y, dy)

#Limits of fitting values
Model.Mu.range(1,100)  
Model.Sigma.range(0,4)
Model.Amplitude.range(0.01,100)  

# Model.Gradient.dev(std=0.05, mean=None, limits=None)
# Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values
Model.Mu.value = 30
Model.Sigma.value = 1
Model.Amplitude.value = 5

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()
