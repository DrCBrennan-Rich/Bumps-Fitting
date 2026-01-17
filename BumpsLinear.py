# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 01:50:56 2025
@author: pycbr
"""
#### Run in the console with: bumps -b --fit=dream --burn=200 --samples=1000 --init=random --export=LinearOutput --session=Linear.h5 BumpsLinear.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

def line(x, Gradient, Intercept):
    return Gradient*x + Intercept

#Load the data from the file Data.txt
x,y,dy = np.loadtxt('Data.txt').T
Model = bmp.Curve(line, x, y, dy)

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
