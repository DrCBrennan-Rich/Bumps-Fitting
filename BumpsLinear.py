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
    """Calculate a linear line from the gradient and intercept.

    Args:
        x (numpy.ndarray): List of (float) x coordinates.
        Gradient (float): Gradient of line.
        Intercept (float): Intercept of line.
        
    Returns:
        y (float): y coordinate of line.

    Notes:
    """
    y = Gradient*x + Intercept
    return y

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
