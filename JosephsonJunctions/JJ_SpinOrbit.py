# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (53) from https://doi.org/10.1103/PhysRevB.55.15174
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_DiffuseStrongExchange.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

k_B = 8.617333262E-5 #eV/K
hbar = 6.582E-16 #eV*s
T_c = 9.2
FreqCutoff=50
JunctionResistance = 1.32E-3

FermiVelocity = 3.3E5*1E9 #nm/s
MeanFreePath = 0.283496 #nm
DiffusionCoeff = FermiVelocity*MeanFreePath/3 #nm^2/s
CoherenceLength = np.sqrt(DiffusionCoeff*hbar/(2*np.pi*k_B*T_c))

N = 1.18363e-13
D = 1.11203e+14 #nm^2/s
tau_SO =  1.58313e-10 #s
ExchangeEnergy = 0.25 #77514
SC_gap = 1.5E-3 #Superconducting gap in eV
Temperature = 4.2 #Temperature in K
d0 = 0

def JC_DiffuseExchange(Thickness,N,D,T_c,tau_SO,ExchangeEnergy,SC_gap,Temperature,d0):
    
    d_F = Thickness- d0
    h = ExchangeEnergy/hbar #Units s^-1
    
    Amplitude = 2*np.pi*N*D*T_c*SC_gap*SC_gap
    
    Alpha = 1/(tau_SO*(np.sqrt(h*h-1/(tau_SO*tau_SO))-h))
               
    RealComponent = 4/(D*tau_SO)
    ImaginaryComponent = 1j*4*np.sqrt(h*h-1/(tau_SO*tau_SO))/D
    
    k_M = np.sqrt(RealComponent + ImaginaryComponent)
    k_M_dag = np.sqrt(RealComponent - ImaginaryComponent)
    
    BracketTerm1 = k_M/np.sinh(k_M*d_F) + k_M_dag/np.sinh(k_M_dag*d_F)
    BracketTerm2 = k_M/np.sinh(k_M*d_F) - k_M_dag/np.sinh(k_M_dag*d_F)
    (2*1j*Alpha)/(1-Alpha*Alpha) 
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff)
    
    Omega_list = (2*N_list +1)*np.pi*k_B*Temperature
    
    for w in Omega_list:
        
        TotalTerm = (1/(w*w))*(BracketTerm1 + (2*1j*Alpha)/(1-Alpha*Alpha)*BracketTerm2)
        
        J_c += TotalTerm 
        
    return Amplitude*np.abs(J_c) #Return the current in milliamps

Load the data from the file Data.txt
d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

OrderingIndex = np.argsort(d)
d = d[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

y = y/1.32E-3
dy = dy/1.32E-3

#d = d + 0.37

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    N = N,
    D= D,
    T_c = T_c,
    tau_SO = tau_SO,
    ExchangeEnergy = ExchangeEnergy,
    SC_gap = SC_gap,
    Temperature = Temperature,
    d0 = d0)

### Limits of fitting values ###

Model.N.range(N*0.1,N*10)
Model.D.range(D*0.1,D*10)
#Model.T_c.range(1,10)
Model.tau_SO.range(tau_SO*0.1,tau_SO*10)
Model.ExchangeEnergy.range(ExchangeEnergy*0.2,ExchangeEnergy*1.8)
#Model.SC_gap.range(30,2000)
#Model.Temperature.range(1.8, 2.5)
Model.d0.range(0,0.5)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values
Model.N.value = N
Model.D.value = D 
Model.T_c.value =T_c
Model.tau_SO.value =tau_SO
Model.ExchangeEnergy.value = ExchangeEnergy
Model.SC_gap.value = SC_gap
Model.Temperature.value = Temperature
Model.d0.value = d0

problem = bmp.FitProblem(Model, constraints=[Model.ExchangeEnergy/hbar*Model.tau_SO > 1])

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot
plt.errorbar(
    d, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.25, 3.0, 1000)

for test in [1.18363e-12]:
    ytest = JC_DiffuseExchange(
        X_axis,
        N = test,
        D= D,
        T_c = T_c,
        tau_SO = tau_SO,
        ExchangeEnergy= ExchangeEnergy,
        SC_gap = SC_gap,
        Temperature = Temperature,
        d0 = d0)
    plt.plot(X_axis, ytest, label=f"Fitted {test}", linewidth=3)

plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=34)
plt.legend(fontsize=34)
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel("Current (mA)", fontsize=34)
plt.show()
