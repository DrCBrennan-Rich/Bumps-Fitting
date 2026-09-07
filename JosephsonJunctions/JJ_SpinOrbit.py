# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (53) from https://doi.org/10.1103/PhysRevB.55.15174
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_DiffuseStrongExchange.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
plt.rcParams.update({'font.size': 40})

k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K
hbar = constants.physical_constants['reduced Planck constant in eV s'][0] #eV*s
T_c = 9.2
FreqCutoff=50

FermiVelocity = 3.3E5*1E9 #nm/s
MeanFreePath = 0.283496 #nm
DiffusionCoeff = FermiVelocity*MeanFreePath/3 #nm^2/s
CoherenceLength = np.sqrt(DiffusionCoeff*hbar/(2*np.pi*k_B*T_c))
JunctionResistance = 1.32E-3 #Ohms

N = 2.18363e-12
D = 1.11203e+14 #nm^2/s
tau_SO =  1.0e-14 #s
ExchangeEnergy = 0.375 #77514   #eV
SC_gap = 1.5E-3 #Superconducting gap in eV
Temperature = 4.2 #Temperature in K
d0 = 0.40

def JC_DiffuseExchange(d_F, N, D, T_c, tau_SO, ExchangeEnergy,
                       SC_gap, Temperature, DeadLayers):
     """Calculate the critical voltage across the Josephson junction according
    to a spin-orbit model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq 53 of the 
    paper by Buzdin et al: https://doi.org/10.1103/PhysRevB.55.15174.

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
    d_F = d_F - DeadLayers
    h = ExchangeEnergy/hbar #Units s^-1
    
    Amplitude = 2*np.pi*N*D*T_c*SC_gap*SC_gap
    
    Alpha = 1/(tau_SO*(np.sqrt((h*h)-1/(tau_SO*tau_SO))-h))
               
    RealComponent = 4/(D*tau_SO)
    ImaginaryComponent = 1j*4*np.sqrt((h*h)-1/(tau_SO*tau_SO))/D
    
    k_M = np.sqrt(RealComponent + ImaginaryComponent)
    k_M_dag = np.sqrt(RealComponent - ImaginaryComponent)
    
    BracketTerm1 = k_M/np.sinh(k_M*d_F) + k_M_dag/np.sinh(k_M_dag*d_F)
    BracketTerm2 = k_M/np.sinh(k_M*d_F) - k_M_dag/np.sinh(k_M_dag*d_F)
    #(2*1j*Alpha)/(1-Alpha*Alpha) 
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff)
    
    Omega_list = (2*N_list +1)*np.pi*k_B*Temperature
    
    for w in Omega_list:
        
        TotalTerm = (1/(w*w))*(BracketTerm1 + ((2*1j*Alpha)/(1-Alpha*Alpha))*BracketTerm2)
        
        J_c += TotalTerm 
        
    return Amplitude*np.abs(J_c) #Return the current in milliamps



def JC_model(d_F, Amplitude, CoherenceLength_F1, CoherenceLength_F2, d_0pi):
    
    SinTerm = np.sin((d_F-d_0pi)/CoherenceLength_F2)
    
    return Amplitude*(np.exp(-d_F/CoherenceLength_F1)*np.abs(SinTerm))

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

OrderingIndex = np.argsort(d)
d = d[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

y = y/JunctionResistance
dy = dy/JunctionResistance

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
    DeadLayers = DeadLayers)

### Limits of fitting values ###

Model.N.range(N*0.1,N*10)
Model.D.range(D*0.1,D*10)
#Model.T_c.range(1,10)
Model.tau_SO.range(tau_SO*0.1,tau_SO*10)
Model.ExchangeEnergy.range(ExchangeEnergy*0.2,ExchangeEnergy*1.8)
#Model.SC_gap.range(30,2000)
#Model.Temperature.range(1.8, 2.5)
Model.DeadLayers.range(-0.5,0.0)

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
Model.DeadLayers.value = DeadLayers

problem = bmp.FitProblem(Model, constraints=[Model.ExchangeEnergy/hbar*Model.tau_SO > 1])

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot
plt.errorbar(
    d, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.0, 3.0, 1000)

for test in [1.18363e-12]:
    ytest = JC_DiffuseExchange(
        X_axis,
        N = N,
        D= D,
        T_c = T_c,
        tau_SO = tau_SO,
        ExchangeEnergy= ExchangeEnergy,
        SC_gap = SC_gap,
        Temperature = Temperature,
        DeadLayers = DeadLayers
    )
    plt.plot(X_axis, ytest, label=f"Fitted {test}", linewidth=3)

plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=34)
plt.legend(fontsize=34)
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel("Current (mA)", fontsize=34)
plt.show()

A = ytest = JC_DiffuseExchange(
    0,
    N = N,
    D= D,
    T_c = T_c,
    tau_SO = tau_SO,
    ExchangeEnergy= ExchangeEnergy,
    SC_gap = SC_gap,
    Temperature = Temperature,
    d0 = d0
)
CoherenceLength_F1 = 0.18
CoherenceLength_F2 = 0.34
d_0pi = 0.457

for test in [1.18363e-12]:
    ytest = JC_model(X_axis, 
                     Amplitude = A, 
                     CoherenceLength_F1 = CoherenceLength_F1, 
                     CoherenceLength_F2= CoherenceLength_F2, 
                     d_0pi = d_0pi)
    plt.plot(X_axis, ytest, label="Phenomenlogical", linewidth=3)

plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=34)
plt.legend(fontsize=34)
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel("Current (mA)", fontsize=34)
plt.show()
