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
CoherenceLength = 3.0 #nm
Resistivity = 16.8 #ohm nm
e = 1#1.6021766E-19 #Coulombs
hbar = 6.582E-16 #eV*s
H =1 
tau = 1

n=1
T = 4.2
T_c = 8.5
FreqCutoff=50

Area = np.pi*(1.5E3)*(1.5E3) #Area of the gate in nm

def Trancendental_Quartic(Chi_vec,gamma,Omega,eta,theta):
    Chi = Chi_vec[0]+1j*Chi_vec[1]
    S = np.sin(theta)
    u = np.sqrt(Omega+eta*(1-Chi*Chi))
    Residual = Chi**4+(2*gamma*u*S)*Chi**3+((gamma*u)**2-1)*Chi**2-(gamma*u*S)*Chi+0.25*S*S
    return [np.real(Residual), np.imag(Residual)]

def Solve_Quartic_Exact(gamma,Omega,theta):
    S = np.sin(theta)
    u = np.sqrt(Omega)

    coeffs = [1,2*gamma*u*S,(gamma*u)**2-1,-(gamma*u*S),0.25*S*S]

    Roots = np.roots(coeffs)
    return Roots

def Pick_Root(Roots,gamma,Omega,theta):
    
    LHS = 2*gamma*np.sqrt(Omega)*Roots
    RHS = np.sin(theta-2*np.arcsin(Roots))
    
    i = np.argmin(np.abs(RHS-LHS))
    return Roots[i]

def solve_chi_continuation(gamma, Omega, theta, eta):
    #Initial value taken from exact solution when eta=0
    Roots = Solve_Quartic_Exact(gamma, Omega, theta)
    chi0 = Pick_Root(Roots,gamma,Omega,theta)

    Guess = [chi0.real, chi0.imag]
    #Relax eta=0 condition
    Solution = fsolve(
        Trancendental_Quartic,
        x0=Guess,
        args=(gamma, Omega, eta, theta)
    )

    return Solution[0] + 1j*Solution[1]


def JC_DiffuseExchange(d_F, Temperature, Resistivity, theta, eta, CoherenceLength, H):
    
    Amplitude = Area*(16*np.pi*k_B*Temperature)/(Resistivity)
    
    J_c = np.zeros_like(d_F, dtype='float')
    
    N_list = np.arange(FreqCutoff)
    Omega_list = (T/T_c)*(2*N_list+1)+(H/(np.pi*k_B*T_c))*1j

    eta = hbar/(np.pi*tau*k_B*T_c)
    
    gamma_list = np.sqrt(Omega_list+eta)/CoherenceLength
    
    for gamma, w in zip(gamma_list, Omega_list):
        
        theta = np.arctan(SC_gap/(k_B*T_c*w))
        Chi = solve_chi_continuation(gamma, w, theta, eta)

        Term = np.real(gamma*np.exp(-gamma*d_F)*Chi*Chi)
        J_c += Term
         
    return Amplitude*np.abs(J_c)

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    Temperature=Temperature,
    Resistivity = Resistivity,
    H=H)

### Limits of fitting values ###

Model.CoherenceLength.range(1E-3,10)
Model.H.range(1E-6,5E-3)
#Model.Temperature.range(3,5)
#Model.Temperature.value = 4.2
Model.eta.range(1.3E-6,1.5E-1)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######

#Initial values

Model.CoherenceLength.value = 1.3
Model.H.value = 1.5E-3
Model.Temperature.value = 4.2
Model.eta.value = 1.4E-3

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

for H_test in [0.5E2,2E2,5E2]:
    ytest = JC_DiffuseExchange(
        d,
        Temperature=Temperature,
        Resistivity = Resistivity,
        CoherenceLength=0.58,
        theta=1,
        eta=0.04,
        H=H_test)
    plt.plot(d, ytest, label=f"H_test={H_test}")

plt.legend()
plt.yscale('log')
plt.savefig("BallisticSimplified.svg", format="svg")
plt.show()
