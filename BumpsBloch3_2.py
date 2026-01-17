# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 01:50:56 2025
@author: pycbr
"""
#### Run with: bumps -b --fit=dream --burn=200 --samples=1000 --init=random --export=BlochOutput --session=BlochSession.h5 BumpsBloch3_2.py

import bumps.names as bmp
import numpy as np

def BlochLaw(x, M0, b):
    
    return M0*(1 - b*x**(1.5))

x,y,dy = np.loadtxt('BlochData.txt').T
Model = bmp.Curve(BlochLaw, x, y, dy)

#Limits of fitting values
Model.M0.range(1,1000)  
Model.b.range(0,1)  

# Model.Gradient.dev(std=0.05, mean=None, limits=None)
# Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values
Model.M0.value = 500
Model.b.value = 0.2

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()
