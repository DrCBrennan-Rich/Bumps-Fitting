# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 01:50:56 2025

@author: pycbr
"""

#### Can be ran in console with: python -m bumps.cli --fit=dream --burn=2000 --samples=1e5 --init=eps --parallel --store=output BumpsLinear.py

import bumps.names as bmp
import numpy as np
import matplotlib as plt
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

problem.show()

result = bmp.fit(problem, 
                 method='dream',
                 burn=2000,
                 samples=1e4,
                 store = 'New_output' )

print('The Gradient is',result['x'][0],'with error', result['dx'][0])
print('The Intercept is',result['x'][1],'with error', result['dx'][1])

#Errors = errplot.calc_errors_from_state(problem, state, nshown=50, random=True, portion: float | None = None)
#errplot.show_errors(Errors)

'''
problem.show()



problem.show()

#state = result.state
#samples = result.state.draw()
#np.save("New_Output_2.npy", samples)

state = result.state
chains = state.chains

chains = np.asarray(state.chains)

# drop burn-in (e.g. first 20%)
burn = int(0.2 * chains.shape[1])
chains = chains[:, burn:, :]

# flatten chains
samples = chains.reshape(-1, chains.shape[-1])

lo, mid, hi = np.percentile(samples, [16, 50, 84], axis=0)

names = [p.name for p in problem.parameters()]
for n, m, l, h in zip(names, mid, lo, hi):
    print(f"{n:15s} = {m:.5g} (+{h-m:.5g}, -{m-l:.5g})")



'''