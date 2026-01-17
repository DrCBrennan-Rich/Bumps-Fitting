# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 01:50:56 2025

@author: pycbr
"""

#### Can be ran in console with: python -m bumps.cli --fit=dream --burn=2000 --samples=1e5 --init=eps --parallel --store=output BumpsLinear.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from bumps import errplot


def line(x, Gradient, Intercept):
    return Gradient*x + Intercept

x,y,dy = np.loadtxt('Data.txt').T
Model = bmp.Curve(line, x, y, dy)

#Limits of fitting values
Model.Gradient.range(0,1)  
Model.Intercept.range(0,1)  

# Model.Gradient.dev(std=0.05, mean=None, limits=None)
# Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values
Model.Gradient.value = 0.2
Model.Intercept.value = 0.5

problem = bmp.FitProblem(Model)
'''
problem.show()


result = bmp.fit(problem, 
                method='dream',
                burn=2000,
                samples=1e3,
                store = "Users\pycbr\Documents\PhD\Spyder\Output2")


#Extracting the data
Draw = result.state.draw()    
Samples = Draw.points 

lower, median, upper = np.percentile(Samples, [16, 50, 84], axis=0)
GradientErrors = [median[0]-lower[0],upper[0]-median[0]]
InterceptErrors = [median[1]-lower[1],upper[1]-median[1]]

print('The Gradient is',result['x'][0],'with error range', GradientErrors)
print('The Intercept is',result['x'][1],'with error range', InterceptErrors)
'''

'''
GradientSamples  = [A[0] for A in Samples]
InterceptSamples  = [B[1] for B in Samples]

Counts1, Bin_edges1 = np.histogram(GradientSamples, bins=50)
Counts2, Bin_edges2 = np.histogram(InterceptSamples, bins=50)

plt.hist(GradientSamples, bins=50)
plt.xlabel("Gradient")
plt.ylabel("Count")

plt.show()

plt.hist(InterceptSamples, bins=50)
plt.xlabel("Intercept")
plt.ylabel("Count")

plt.show()

'''

#Errors = errplot.calc_errors_from_state(problem, state, nshown=50, random=True, portion: float | None = None)
#errplot.show_errors(Errors)

