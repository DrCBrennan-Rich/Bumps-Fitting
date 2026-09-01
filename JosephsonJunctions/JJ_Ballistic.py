# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (1) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_Ballistic.py
'''
#### WARNING ####

The function JC_model has been written to be fast and easily callable thousands
of times. This is at the cost of stability if the ferromagnetic coherence
length is smaller than 0.2 nm (see associated Figure on GitHub):
https://github.com/DrCBrennan-Rich/Bumps-Fitting/blob/main/JosephsonJunctions/Ballistic_CoherenceLengthComparison.pdf
And so this should be set as the lower fitting bound.
In situations where the coherence length is approaching this limit, it would be advisable to use:
https://github.com/DrCBrennan-Rich/Bumps-Fitting/blob/main/JosephsonJunctions/JJ_Ballistic_simplified.py
'''

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy import constants

k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K

T = 5 #Temperature in K
alpha = 1
y_max = 10*alpha
PhiIncriment = 60

JunctionResistance = 1.4E-3 #ohms
T = 5 #K
SC_gap = 1.5E-3 #eV

Beta = 1/(k_B*T)

def JC_model(d_F, CoherenceLength, SC_gap, PhiIncriment):
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
    
    #Produce a list of phase differences across the junction.
    PhiList = np.linspace(0, 2*np.pi, PhiIncriment, endpoint=False)[:,None]
    
    #Produces a list of alpha values with minumum value 1E-6
    AlphaList = np.maximum(d_F/CoherenceLength, 1E-6)
    
    IcRn = np.zeros_like(AlphaList)
    
    for i,Alpha in enumerate (AlphaList):
        
        #The upper limit of the integral either 10*lower limit or 5
        yMax = max(10*Alpha, 5)
        #Produce the step number with minimum value 200
        StepNumber = max(200, int(100*yMax))
        yList = np.linspace(Alpha, yMax, StepNumber)[None,:]
        
        SinMinus = np.sin(0.5*(PhiList-yList))
        SinPlus = np.sin(0.5*(PhiList+yList))
        
        TanhMinus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList - yList)))
        TanhPlus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList + yList)))
        
        Integrand = (1/(yList*yList*yList))*(SinMinus*TanhMinus+SinPlus*TanhPlus)
        
        Iphi = np.pi*SC_gap*Alpha*Alpha*trapezoid(Integrand, x=yList, axis=1)/2
        
        IcRn[i] = np.max(np.abs(Iphi))
        
    return 1E-6*IcRn #Voltage in uV
        

#Load the data from the file Data.txt
#d,y,dy = np.loadtxt('L11 data 4.2K.txt').T

Model = bmp.Curve(JC_model, d, y, dy)

### Limits of fitting values ###

Model.SC_gap.range(1,50) 
Model.CoherenceLength.range(0.001,100)  

#Model.Gradient.dev(std=0.05, mean=None, limits=None)
#Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values

Model.SC_gap.value = 31 #eV
Model.CoherenceLength.value = 0.2

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Plot data points if desired

plt.errorbar(
    d, y, yerr=dy,
    fmt='o',
    capsize=3,
    label='Experimental data'
)

x_axis = np.linspace(0.01,10,100)

for CoherenceLength_test in [5E-1]:
    ytest = JC_model(
        x_axis,
        CoherenceLength=CoherenceLength_test,
        SC_gap = SC_gap)
    
    plt.plot(x_axis, ytest, label=f"Coherence Length={CoherenceLength_test}")
    plt.yscale('log')

plt.xlabel("Ferromagnet thickness (nm)")
plt.ylabel(r"Critical current density $J_c$ (mA/m$^2$)")

plt.legend()
plt.savefig("CoherenceLengthComparison.svg", format="svg")
plt.show()
