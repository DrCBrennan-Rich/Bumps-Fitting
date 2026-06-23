# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (1) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJCriticalCurrentHeim.py

import bumps.names as bmp
import numpy as np


T = 5 #Temperature in K
T_c = 10 #Critical temperature in K
alpha = 1
y_max = 10*alpha

def IcRn_single(d, xi, Delta):
    kB = 1#1.38e-23
    e = 1#1.6e-19
    T = 5

    #Precompute on grid
    alpha = max(d/xi, 1e-6)
    y_max = 10*alpha
    Ny = 300
    y = np.linspace(alpha, y_max, Ny)

    PhiList = np.linspace(0, 2*np.pi, 100)

    A = np.pi*Delta*alpha*alpha/(2*e)

    Ic = 0

    for phi in PhiList:
        Sin1 = np.sin(0.5*(phi-y))
        Sin2 = np.sin(0.5*(phi+y))

        Tanh1 = np.tanh(Delta*np.cos(0.5*(phi-y))/(2*kB*T))
        Tanh2 = np.tanh(Delta*np.cos(0.5*(phi+y))/(2*kB*T))

        Integrand = (Sin1*Tanh1+Sin2*Tanh2)/(y*y*y)

        Iphi = A*np.trapezoid(Integrand, y)

        if Iphi > Ic:
            Ic = Iphi

    return Ic

def IcRn_model(d, xi, Delta):
    return np.array([IcRn_single(di, xi, Delta) for di in np.atleast_1d(d)])


#Load the data from the file Data.txt
d,y,dy = np.loadtxt('TestData2.txt').T 
Model = bmp.Curve(IcRn_model, d, y, dy)

### Limits of fitting values ###

Model.Delta.range(25,30) 
Model.xi.range(0.01,7)  

#Model.Gradient.dev(std=0.05, mean=None, limits=None)
#Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values

Model.Delta.value = 31
Model.xi.value = 0.2

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()
