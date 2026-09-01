# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (2) from https://doi.org/10.1063/5.0195229
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_Ballistic_simplified.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K

SC_gap = 1.5E-3 #eV
Temperature = 4.2 #K
CoherenceLength = 0.3 #nm
JunctionResistance = 1.4E-3 #ohms

def JC_model(d_F, Temperature, SC_gap, CoherenceLength, Amplitude):
    """Calculate the critical voltage across the Josephson junction according
    to a simplifed ballistic model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq. 2 of 
    the paper by Birge and Satchell: https://doi.org/10.1063/5.0195229.
    It is valid if the ferromagnetic thickness is significantly larger than the
    feromagnetic coherence length and for temperatures near the critical 
    temperature of the superconductor.

    Args:
        d_F (numpy.ndarray): List of (float) thicknesses of the ferromagnetic 
            junction (nm).
        SC_gap (float): Superconducting gap (eV).
        CoherenceLength (float): Coherence length in the ferromagnet (nm).
        Amplitude (float): If provided, can be used to set an arbitrary scaled
            amplitude for the output.

    Returns:
        IcRn (float): Voltage across the Josephson junction (uV).

    Notes:
        Equation being solved is IcRn = pi*SC_gap^2*Sinc[d_F/CoherenceLength]/(4*T)
    """

    if Amplitude is None:
        Amplitude = (np.pi*SC_gap*SC_gap)/(4*k_B*Temperature)
        
    #the np.sinc function is normalised as default so need to divide argument by pi
    SincTerm = np.sinc(d_F/(np.pi*CoherenceLength)) 
    
    return Amplitude*np.abs(SincTerm) 

#Load the data from the file Data.txt
#d,y,dy = np.loadtxt('L11 data 4.2K.txt').T #units of nm, mA, mA

Model = bmp.Curve(
    JC_model,
    d, y, dy,
    Temperature=Temperature,
    SC_gap = SC_gap,
    CoherenceLength=CoherenceLength, 
    Resistance=Resistance)

### Limits of fitting values ###

Model.CoherenceLength.range(1,3)
Model.SC_gap.range(1E-3,2E-3)
Model.Temperature.range(3,5)
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

X_axis = np.linspace(0.1, 2.5, 1000)

for CoherenceLenght_test in [0.1,0.5,1,3]:
    ytest = JC_model(
        X_axis,
        Temperature=Temperature,
        SC_gap = SC_gap,
        CoherenceLength=CoherenceLength, 
        Resistance=Resistance, 
    )
    plt.plot(X_axis, ytest, label=f"CoherenceLength={CoherenceLenght_test}")

plt.legend()
plt.yscale('log')
plt.savefig("BallisticSimplified.svg", format="svg")
plt.show()
