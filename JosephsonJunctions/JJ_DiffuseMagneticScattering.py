# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (A10) from https://link.aps.org/doi/10.1103/RevModPhys.77.935
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_DiffuseMagneticScattering.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
plt.rcParams.update({'font.size': 40})

#Physical constants
k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K
hbar = constants.physical_constants['reduced Planck constant in eV s'][0] #eV*s

#Function parameters
FreqCutoff=50
StepNumber = 5

#Potential fitting values
T_c = 9.2
Temperature=4.2 #K
H = 0.4  #Exchange energy in eV
alpha =  1.03595
SC_gap = 1.5E-3 #eV
CoherenceLength = 0.711048
d0 = 0.488447

def JC_MagneticScattering(d_F, Temperature, T_c, H, SC_gap, alpha, 
                          CoherenceLength, DeadLayers, Amplitude=None):
    """Calculate the critical voltage across the Josephson junction according
    to a magnetic scattering model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq A10 of the 
    paper by Buzdin et al: https://link.aps.org/doi/10.1103/RevModPhys.77.935.

    Args:
        d_F (numpy.ndarray): List of (float) thicknesses of the ferromagnetic junction (nm).            
        Temperature (float): Temperature of the system (K).
        T_c (float): Critical temperature of the superconductor (K).       
        H (float): Exchange energy in the ferromagnet (eV).        
        SC_gap (float): Superconducting gap (eV).       
        alpha (float): Magnetic scattering parameter defined as 1/(h*tau_s) (unitless).        
        CoherenceLength (float): Coherence length in the ferromagnet (nm).      
        DeadLayer (float): Thickness of dead (non-magnetic) material in the ferromagnet. 
            Negative values indicate increased effective ferromagnetic thickness 
            due to proximity magnetisation in the normal metal (nm).         
        Amplitude (float): If provided, can be used to set an arbitrary scaled
            amplitude for the output (unitless).
            
    Returns:
        IcRn (float): Critical voltage of the Josephson junction (uV).

    Notes:
    """
    
    ThicknessEffective = d_F - DeadLayers
    d_F = ThicknessEffective/CoherenceLength
    h = H/hbar #s^-1
    
    if Amplitude is None:
        Amplitude = 64*np.pi*Temperature/3  
    
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff) #List of integers
    
    Omega_list = (2*N_list +1)*np.pi*k_B*Temperature/hbar #Matsurbara frequences, s^-1

    q_list = np.sqrt(2j+2*alpha+2*Omega_list/h) 

    OMEGA_list = np.sqrt(Omega_list*Omega_list+SC_gap*SC_gap/(hbar*hbar))

    Phi_list = SC_gap*SC_gap/(hbar*hbar*(OMEGA_list+Omega_list)*(OMEGA_list+Omega_list))
    
    eta_list = np.sqrt(alpha/(alpha+1j+Omega_list/h))
     
    for q, phi, eta in zip(q_list,Phi_list,eta_list):
        
        Numerator = 2*q*d_F*phi*np.exp(-2*q*d_F)
        
        Denominator = np.sqrt((1-eta*eta)*phi+1)+1
        
        Term = np.real(Numerator/(Denominator*Denominator))
      
        J_c += Term
    
    IcRn = Amplitude*np.abs(J_c)
        
    return  IcRn*1E6 #Return the critical voltage in uV

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
    alpha = alpha,
    CoherenceLength = CoherenceLength,
    d0 = d0)

### Limits of fitting values ###

#Model.Temperature.range(0.9*Temperature, 1.1*Temperature)
#Model.T_c.range(0.5*T_c,1.5*T_c)
#Model.H.range(0.2, 0.6)
#Model.SC_gap.range(0.5*SC_gap, 1.5*SC_gap)
Model.alpha.range(0.01*alpha, 100*alpha)
Model.CoherenceLength.range(CoherenceLength*0.01, 10*CoherenceLength)
Model.d0.range(d0*0.00, 10*d0)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.H.dev(std=0.1, mean=4, limits=None)

#######
#Initial values

Model.Temperature.value = Temperature
Model.T_c.value = T_c 
Model.H.value = H
Model.SC_gap.value = SC_gap
Model.alpha.value = alpha
Model.CoherenceLength.value = CoherenceLength
Model.d0.value = d0

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

X_axis = np.linspace(0.1, 2.5, 1000)

for test in [0.4]:
    ytest = JC_MagneticScattering(
        X_axis, 
        Temperature = Temperature, 
        T_c = T_c, 
        H = H, 
        SC_gap = SC_gap, 
        alpha = alpha,
        CoherenceLength = CoherenceLength,
        d0 = d0)
    plt.plot(X_axis, ytest, label="Fitted Curve", linewidth=3)
    

plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=34)
plt.legend(fontsize=34)
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel("Current (mA)", fontsize=34)
plt.show()
