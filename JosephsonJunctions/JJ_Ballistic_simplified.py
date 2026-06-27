# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (2) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_Ballistic_simplified.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

k_B = 8.617333262E-5 #eV/K
SC_gap = 1.5E-3 #eV
T = 4.2 #K
xi = 0.3 #nm
R = 1.4E-3 #ohms
#Coherence lengths
CoherenceLength_F1=0.3 #nm
CoherenceLength_F2=0.16 #nm

def JC_model(d_F, Temperature, SC_gap, CoherenceLength, Resistance):
    
    SincTerm = np.sin(d_F/CoherenceLength)/(d_F/CoherenceLength)
    
    return 1E3*(np.pi*SC_gap*SC_gap)*np.abs(SincTerm)/(4*k_B*Temperature)

#Load the data from the file Data.txt
d_F,y,dy = np.loadtxt('L11 data 4.2K.txt').T 

Model = bmp.Curve(
    JC_model,
    d_F, y, dy,
    Temperature=T,
    SC_gap = SC_gap,
    CoherenceLength=xi, 
    Resistance=R)

### Limits of fitting values ###

#Model.Amplitude.range(20,120)
#Model.CoherenceLength_F1.dev(std=0.1, mean=0.3, limits=None)

#Model.d_0pi.range(0.0,0.9*np.pi*CoherenceLength_F2) #Due to the periodicity of d_0pi, this will be the paramter range

#Model.CoherenceLength.range(0.1,0.3)
#Model.CoherenceLength.range(0.1,0.2)

#Model.CoherenceLength_F1.dev(std=0.1, mean=0.3, limits=None)
#Model.CoherenceLength_F2.dev(std=0.1, mean=0.16, limits=None)

#######

#Initial values

#Model.Amplitude.value = 130
#Model.d_0pi.value = 1

#Model.CoherenceLength.value = 0.1
#Model.CoherenceLength.value = 0.16

#problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
#problem.show()

#Run some test values to see how they affect the final plot


plt.errorbar(
    d_F, y, yerr=dy,
    fmt='o',
    capsize=3,
    label='Experimental data'
)


for CoherenceLenght_test in [0.5]:
    ytest = JC_model(
        d_F,
        Temperature=T,
        SC_gap = SC_gap,
        CoherenceLength=CoherenceLenght_test, 
        Resistance=R, 
    )
    plt.plot(d_F, ytest, label=f"CoherenceLength={CoherenceLenght_test}",color='red')

plt.legend()
plt.yscale('log')
plt.savefig("BallisticSimp.svg", format="svg")
plt.show()
