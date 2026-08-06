# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (A10) from https://link.aps.org/doi/10.1103/RevModPhys.77.935
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_DiffuseMagneticScattering.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

k_B = 8.617333262E-5 #eV/K
hbar = 6.582E-16 #eV*s

FreqCutoff=50
StepNumber = 5
T_c = 9.2

FermiVelocity = 3.3E5*1E9 #nm/s
MeanFreePath = 0.283496 #nm
DiffusionCoeff = FermiVelocity*MeanFreePath/3 #nm^2/s
CoherenceLength = np.sqrt(DiffusionCoeff*hbar/(2*np.pi*k_B*T_c))

Temperature=4.2 #K
H = 0.679
alpha = 1
SC_gap = 1.5E-3 #eV


def JC_MagneticScattering(d_F, Temperature, T_c, H, SC_gap, alpha):
    
    h = H/hbar
    Amplitude = 64*np.pi*Temperature/3
    
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff)
    #
    Omega_list = (Temperature/T_c)*(2*N_list+1)+(h/(np.pi*k_B*T_c))*1j

    q_list = np.sqrt(2j+2*alpha+2*Omega_list/h)

    OMEGA_list = np.sqrt(Omega_list*Omega_list+SC_gap*SC_gap)

    Phi_list = SC_gap*SC_gap/((OMEGA_list+Omega_list)*(OMEGA_list+Omega_list))
    
    eta_list = np.sqrt(alpha/(alpha+1j+Omega_list/h))
     
    
    for q, phi, eta in zip(q_list,Phi_list,eta_list):
        
        Numerator = 2*q*d_F*phi*np.exp(2*q*d_F)
        
        Denominator = np.sqrt((1-eta*eta)*phi+1)+1
        
        Term = np.real(Numerator/(Denominator*Denominator))
      
        J_c += Term
        
    return Amplitude*np.abs(J_c) #Return the current in milliamps

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

OrderingIndex = np.argsort(d)
d = d[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

#y = y/1.9E-3
#dy = dy/1.9E-3

Model = bmp.Curve(
    JC_MagneticScattering, 
    d, y, dy,
    Temperature = Temperature, 
    T_c = T_c, 
    H = H, 
    SC_gap = SC_gap,
    alpha = alpha)

### Limits of fitting values ###

Model.Temperature.range = Temperature, 
Model.T_c.range = [0.5*T_c,1.5*T_c] 
Model.H.range = [0.5*H, 1.5*H]
Model.SC_gap.range = [0.5*SC_gap, 1.5*SC_gap]
Model.alpha.range = [0.5*alpha, 1.5*alpha]

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values

Model.Temperature.value = Temperature
Model.T_c.value = T_c 
Model.H.value = H
Model.SC_gap.value = SC_gap
Model.alpha.value = alpha

#JC_DiffuseExchange(d_F, Temperature, Resistivity, SpinScatterTime, CoherenceLength, H, gamma_NF, gamma_BSN, d_N, xi_N)

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot

plt.errorbar(
    d, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.1, 1.4, 1000)

for test in [30]:
    ytest = JC_MagneticScattering(
        X_axis, 
        Temperature = Temperature, 
        T_c = T_c, 
        H = H, 
        SC_gap = SC_gap, 
        alpha = alpha)
    
plt.plot(X_axis, ytest, label=f"Fitted", linewidth=3)
plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=34)
plt.legend(fontsize=34)
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel("Current (mA)", fontsize=34)
plt.show()
