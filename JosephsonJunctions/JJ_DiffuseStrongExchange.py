# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (18) from https://doi.org/10.1088/1367-2630/17/11/113022
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_DiffuseStrongExchange.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

k_B = 8.617333262E-5 #eV/K
SC_gap = 1.5E-3 #eV
Temperature = 4.2 #K
CoherenceLength = 0.3 #nm
Resistivity = 1.4E-3 #ohm meters
e = 1.6021766E-19 #Coulombs

def JC_DiffuseExchange(d_F, Temperature, SC_gap, CoherenceLength, Resistance):
#the np.sinc function is normalised as default so need to divide argument by pi
    SincTerm = np.sinc(d_F/(np.pi*CoherenceLength)) 
    
    return (np.pi*SC_gap*SC_gap)*np.abs(SincTerm)/(4*k_B*Temperature*Resistance)

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('L11 data 4.2K.txt').T #units of nm, mA, mA

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    Temperature=Temperature,
    SC_gap = SC_gap,
    CoherenceLength=CoherenceLength, 
    Resistance=Resistance)

### Limits of fitting values ###

#Model.CoherenceLength.range(1,3)
Model.SC_gap.range(1E-3,2E-3)
Model.Temperature.range(3,5)
#Model.Temperature.value = 4.2
#Model.Resistance.range(1.3E-3,1.5E-3)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######

#Initial values

Model.CoherenceLength.value = 0.3
Model.SC_gap.value = 1.5E-3
Model.Temperature.value = 4.2
#Model.Resistance.value = 1.4E-3

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot
plt.errorbar(
    d, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data'
)

for CoherenceLenght_test in [0.5]:
    ytest = JC_DiffuseExchange(
        d,
        Temperature=Temperature,
        SC_gap = SC_gap,
        CoherenceLength=CoherenceLenght_test, 
        Resistance=Resistance, 
    )
    plt.plot(d, ytest, label=f"CoherenceLength={CoherenceLenght_test}")

plt.legend()
plt.yscale('log')
plt.savefig("BallisticSimplified.svg", format="svg")
plt.show()
