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

Resistivity = 16.8 #ohm nm
hbar = 6.582E-16 #eV*s
FermiVelocity = 3.3E5*1E9 #nm/s
MeanFreePath = 0.283496 #nm
DiffusionCoeff = FermiVelocity*MeanFreePath/3 #nm^2/s
T_c = 8.5
CoherenceLength = np.sqrt(DiffusionCoeff*hbar/(2*np.pi*k_B*T_c))
AR = 5.7*1E3 #Ohm nm^2
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
    
    EtaSteps = np.linspace(0,eta,10)
    Guess = [chi0.real, chi0.imag]
    
    for EtaIntermediate in EtaSteps:
        #Relax eta=0 condition
        Solution = fsolve(
            Trancendental_Quartic,
            x0=Guess,
            args=(gamma, Omega, EtaIntermediate, theta)
        )
        Guess = [Solution[0], Solution[1]]

    return Solution[0] + 1j*Solution[1]


def JC_DiffuseExchange(d_F, Temperature, Resistivity, SpinScatterTime, CoherenceLength, H):
    
    Amplitude = Area*(16*np.pi*k_B*Temperature)/(Resistivity)
    
    J_c = np.zeros_like(d_F, dtype='float')
    
    N_list = np.arange(FreqCutoff)
    Omega_list = (Temperature/T_c)*(2*N_list+1)+(H/(np.pi*k_B*T_c))*1j

    eta = hbar/(np.pi*SpinScatterTime*k_B*T_c)
    gamma_BNF = AR/(CoherenceLength*Resistivity)
    gamma_list = np.sqrt(Omega_list+eta)/CoherenceLength
    
    for gamma, w in zip(gamma_list, Omega_list):
        
        theta = np.arctan(SC_gap/(k_B*T_c*w))
        
        Chi = solve_chi_continuation(gamma_BNF, w, theta, eta)

        Term = np.real(gamma*np.exp(-gamma*d_F)*Chi*Chi)
        J_c += Term
         
    return Amplitude*np.abs(J_c)

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    Temperature=Temperature,
    Resistivity = Resistivity)

### Limits of fitting values ###

#Model.CoherenceLength.range(1E-3,10)
Model.H.range(1E-6,5E-3)
#Model.Temperature.range(1,10)
Model.SpinScatterTime.range(1E-14,1E-8)
Model.Resistivity.range(10,1000)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######

#Initial values

Model.CoherenceLength.value = CoherenceLength
Model.H.value = 1.5E-3
Model.Temperature.value = 4.2
Model.SpinScatterTime.value = 100E-15
Model.Resistivity.value = 62

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

for tau_test in [100E-15]: #0.00432
    ytest = JC_DiffuseExchange(
        d,
        Temperature=Temperature,
        Resistivity = Resistivity,
        CoherenceLength=0.3,
        gamma_BNF = 0.5,
        SpinScatterTime= tau_test,#0.109,
        H=0.01)
    plt.plot(d, ytest, label=f"eta={tau_test}")
    
    
plt.legend()
plt.yscale('log')
plt.savefig("DiffuseStrongExchange.svg", format="svg")
plt.show()
