# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (27) from 10.1088/1367-2630/17/11/113022
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=DiffuseFit --session=JJSession.h5 JJ_Diffuse.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

k_B = 8.617333262E-5  #eV/K
hbar = 6.582E-16 #eV*s
T = 4.2    #K Temperature
T_c = 9.2  #K Critical temperature

#Coherence lengths
CoherenceLength_F = 30.0 #nm
CoherenceLength_N = 30.0 #nm

nm = 1E-9 #Conversion factor

#Load the data from the file Data.txt
Data = np.loadtxt("L11 data 4.2K.txt")
Thicknesses = Data[:, 0]*nm #Thickness in m

dN = 100*nm   #N thickness

#Diffusion Coefficients defined as CoherenceLength^2 * (2pi*k_B*T_c)/hbar
DiffusionCoef_N = 2*np.pi*k_B*T_c*(CoherenceLength_N*nm)**2/hbar #m^2/s

DiffusionCoef_F = 2*np.pi*k_B*T_c*(CoherenceLength_F*nm)**2/hbar

#Diffusion constants, if using dirty limit: D ~xi^2*(pi*k_B*T_c/hbar).
#DN = (CoherenceLength_N/CoherenceLength_F)*(CoherenceLength_N/CoherenceLength_F) #1e-2
#DF = 1E0 #1E-5

SC_gap = 1.55E-3 #eV

Delta = 1.55E-3 #eV

#Matsubara cutoff frequency
FreqCutoff=5

def JC_model2(d_F, Amplitude, H, dN, T, 
              DiffusionCoef_N, DiffusionCoef_F, 
              SC_gap, nmax=FreqCutoff):
  
    dF = np.asarray(Thicknesses, dtype=float)
    jc = np.zeros_like(dF)
    
    d_F = np.asarray(Thicknesses, dtype=float)
    jc = np.zeros_like(d_F)
    
    for n in range(int(nmax)):
        omega = np.pi*k_B*T*(2*n+1)/hbar #s^-1

        # diffusion wavevectors
        WaveVec_N = np.sqrt(2*omega/DiffusionCoef_N)
        WaveVec_F = np.sqrt(2*(omega+H*1j)/DiffusionCoef_F)

        z_N = WaveVec_N*dN
        z_F = WaveVec_F*d_F

        #sinh_zF = 0.5*(exp_pos-exp_neg)

        sinh_inv = 1.0/np.sinh(z_F)

        #Stable sech^2(x) = 1 / cosh^2(x)
        sech2 = 1.0/(np.cosh(z_N)*np.cosh(z_N))

        Term = ((SC_gap*SC_gap)/(omega*omega))*(sech2)*np.real(WaveVec_F*sinh_inv)

        jc += Term

    return Amplitude*np.abs(jc)/1E-18


d,y,dy = Data.T 
#d = d/CoherenceLength_F


Model = bmp.Curve(
    JC_model2,
    d, y, dy,
    Amplitude=1.0,
    H=500,
    dN=dN,
    T=4.2/T_c,
    DiffusionCoef_N=DiffusionCoef_N,
    DiffusionCoef_F=DiffusionCoef_F,
    SC_gap = Delta
)

### Limits of fitting values ###

Model.H.range(10E18,30E19)
Model.Amplitude.range(25,300)

#Model.Gradient.dev(std=0.05, mean=None, limits=None)
#Model.Intercept.dev(std=0.5, mean=None, limits=None)

#######

#Initial values

Model.H.value = 15E18
Model.Amplitude.value = 100

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

plt.errorbar(
    d, y, yerr=dy,
    fmt='o',
    capsize=3,
    label='Experimental data'
)

for Htest in [40E14]:
    ytest = JC_model2(
        d,
        Amplitude=1,
        H=Htest,
        dN=dN,
        T=T/T_c,
        DiffusionCoef_N=DiffusionCoef_N,
        DiffusionCoef_F=DiffusionCoef_F,
        SC_gap = Delta,
        nmax=FreqCutoff
    )
    plt.plot(d, ytest, label=f"H={Htest}")
plt.yscale('log')
plt.legend()
plt.show()


for A_test in [1,3,5,10]:
    ytest = JC_model(
        d,
        Amplitude=A_test,
        H=4.5E17,
        dN=dN,
        T=T/T_c,
        DiffusionCoef_N=DiffusionCoef_N,
        DiffusionCoef_F=DiffusionCoef_F,
        SC_gap = Delta,
        nmax=5
    )
    plt.plot(d, ytest, label=f"Amplitude={A_test}")
plt.yscale('log')
plt.legend()
plt.show()
