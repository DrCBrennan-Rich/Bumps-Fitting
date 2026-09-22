# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 01:50:56 2025
@author: pycbr
"""
#### Run in the console with: bumps -b --fit=dream --burn=200 --samples=1000 --init=random --export=LinearOutput --session=Linear.h5 BumpsLinear.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

def Line(x, Gradient, Intercept):
    """Calculate the critical voltage across the Josephson junction according
    to a ballistic model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq. 1 of 
    the paper by Birge and Satchell: https://doi.org/10.1063/5.0195229.

    Args:
        d_F (numpy.ndarray): List of (float) thicknesses of the ferromagnetic 
            junction (nm).
        CoherenceLength (float): Coherence length in the ferromagnet (nm).
        SC_gap (float): Superconducting gap (eV).
        PhiIncriment (int): Number of sub divisions of the 2*pi phase to be 
            tested to find the maxium current (unitless).

    Returns:
        IcRn (float): Voltage across the Josephson junction (uV).

    Notes:
        Equation being solved is IcRn = pi*SC_gap^2*Sinc[d_F/CoherenceLength]/(4*T)
    """
    return Gradient*x + Intercept

#Load the data from the file Data.txt
x,y,dy = np.loadtxt('Data.txt').T
Model = bmp.Curve(Line, x, y, dy)

#Limits of fitting values
Model.Gradient.range(0,3)  
Model.Intercept.range(0,3)  

# Model.Gradient.dev(std=0.05, mean=None, limits=None)
# Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values
Model.Gradient.value = 1.2
Model.Intercept.value = 1.5

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()
