# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (18) from https://doi.org/10.1088/1367-2630/17/11/113022
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_DiffuseStrongExchange.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

k_B = 8.617333262E-5 #eV/K
SC_gap = 1.5E-3 #eV
Temperature = 4.2 #K
CoherenceLength = 0.3 #nm
Resistivity = 1.4E-3 #ohm meters
e = 1.6021766E-19 #Coulombs
hbar = 6.582E-16 #eV*s
H =1 
tau = 1


n=1

T = 4.2
T_c = 8.5

omega = np.pi*k_B*T*(2*n+1)/hbar

Omega = omega/(np.pi*T_c)

OmegaTilde = 1
eta = 1
q = np.sqrt(OmegaTilde+eta)

FreqCutoff=5

N_list = np.linspace(1,FreqCutoff,FreqCutoff)
Omega_list = (T/T_c)*(2*N_list+1)

q_list = np.sqrt(Omega_list+(H/(np.pi*k_B*T_c))+(hbar/(np.pi*tau*k_B*T_c)))

gamma_list = q_list/CoherenceLength

gamma = 1
eta =1
theta = 1

def Trancendental_Quartic(Chi,gamma,Omega,eta,theta):
    
    S = np.sin(theta)
    u = np.sqrt(Omega+eta*(1-Chi*Chi))
    
    return Chi**4+(2*gamma*u*S)*Chi**3+((gamma*u)**2-1)*Chi**2-(gamma*u*S)*Chi+(0.25*S*S)

def JC_DiffuseExchange(d_F, Temperature, Resistivity, gamma_list):
    
    Amplitude = (16*np.pi*Temperature)/(e*Resistivity)
    
    J_c = np.zeros_like(d_F, dtype='float')
    Guess = 0.5
    
    for gamma,omega in zip(gamma_list,Omega_list):
        Chi = fsolve(Trancendental_Quartic, x0=Guess,args=(gamma, Omega, eta, theta))
        Guess = Chi
        
        Term = np.real(gamma*np.exp(-gamma*d_F)*Chi*Chi)
        J_c += Term
         
    return Amplitude*J_c


#Load the data from the file Data.txt
d,y,dy = np.loadtxt('L11 data 4.2K.txt').T #units of nm, mA, mA

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    Temperature=Temperature,
    Resistivity = Resistivity)

### Limits of fitting values ###

#Model.CoherenceLength.range(1,3)
#Model.SC_gap.range(1E-3,2E-3)
Model.Temperature.range(3,5)
#Model.Temperature.value = 4.2
#Model.Resistance.range(1.3E-3,1.5E-3)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######

#Initial values

#Model.CoherenceLength.value = 0.3
#Model.SC_gap.value = 1.5E-3
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
        Resistivity = Resistivity, 
    )
    plt.plot(d, ytest, label=f"CoherenceLength={CoherenceLenght_test}")

plt.legend()
plt.yscale('log')
plt.savefig("BallisticSimplified.svg", format="svg")
plt.show()
