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

FermiVelocity = 3.3E5*1E9 #nm/s
MeanFreePath = 0.283496 #nm
DiffusionCoeff = FermiVelocity*MeanFreePath/3 #nm^2/s
CoherenceLength = np.sqrt(DiffusionCoeff*hbar/(2*np.pi*k_B*T_c))

N = 1E-11
D = 3E13 #nm^2/s
tau_SO = 1E-14 #Spin orbit scatter time in s 
h = 0.8/hbar #Exchange interaction energy in s^-1
SC_gap = 1.5E-3 #Superconducting gap in eV
Temperature = 4.2 #Temperature in K

def JC_DiffuseExchange(d_F,N,D,T_c,tau_SO,h,SC_gap,Temperature):
    
    #Resistivity_F = (Resistivity_N*xi_N)/(gamma_NF*CoherenceLength)
    
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

#Load the data from the file Data.txt
#d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

d = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 1.3, 0.25, 0.3001, 0.35, 0.75, 0.85]) #
y = np.array([np.float64(48.448125000000005), np.float64(17.4225), np.float64(50.02250000000001), np.float64(46.6725), np.float64(46.004000000000005), np.float64(24.02396782574869), np.float64(14.340761412452247), np.float64(20.429403866900394), np.float64(13.615138543432565), np.float64(9.873633936623177), np.float64(7.175099341806733), np.float64(2.3887141453269383), np.float64(4.076677231690765), np.float64(1.4360971319527034), np.float64(26.757749999999998), np.float64(22.880000000000003), np.float64(47.69868888888889), np.float64(11.033598167685819), np.float64(20.58825)]) #
dy = np.array([np.float64(1.6157083779259178), np.float64(1.3225000000000013), np.float64(4.012499999999998), np.float64(7.14), np.float64(2.1280019188587853), np.float64(1.0283281702182476), np.float64(1.52946186871952), np.float64(0.9927739590451741), np.float64(0.9765977782694855), np.float64(0.751572092727746), 0.0868632663427917, np.float64(0.29119788270540087), 0.10592942269344922, np.float64(0.21291291723839048), np.float64(0.4897499999999972), np.float64(0.3199999999999985), np.float64(4.377182207782015), np.float64(0.3490574302442736), np.float64(1.0583323715874893)]) #

OrderingIndex = np.argsort(d)
d = d[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

y = y/1.9E-3
dy = dy/1.9E-3

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    N = N,
    D= D,
    T_c = T_c,
    tau_SO = tau_SO,
    h = h,
    SC_gap = SC_gap,
    Temperature = Temperature)

### Limits of fitting values ###

Model.N.range(1.5,1.9)
#Model.D.range(0.6,0.8)
#Model.T_c.range(1,10)
#Model.tau_SO.range(1,500)
#Model.h.range(0.01,0.1)
Model.SC_gap.range(30,2000)
#Model.Temperature.range(1.8, 2.5)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values

Model.N.value =1
#Model.D.value = 1
#Model.T_c.value =1
#Model.tau_SO.value =1
#Model.h.value = 1
Model.SC_gap.value =1.5E-3
#Model.Temperature.value =1

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot
plt.errorbar(
    d, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.25, 2, 1000)

for test in [0.679]:
    ytest = JC_DiffuseExchange(
        X_axis,
        N = N,
        D= D,
        T_c = T_c,
        tau_SO = tau_SO,
        h = h,
        SC_gap = SC_gap,
        Temperature = Temperature
    )
    plt.plot(X_axis, ytest, label=f"Fitted {test}", linewidth=3)

plt.yscale("linear")
plt.tick_params(axis='both', which='major', labelsize=34)
plt.legend(fontsize=34)
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel("Current (mA)", fontsize=34)
plt.show()
