# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (27) from 10.1088/1367-2630/17/11/113022
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=DiffuseFit --session=JJSession.h5 JJ_Diffuse.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt

k_B = 8.617333262e-5  #eV/K 
T = 4.2    #K Temperature
T_c = 9.2  #K Critical temperature

dF = np.loadtxt("L11 data 4.2K.txt")[:, 0]
dN_nm = 100   #nm N thickness

#Coherence lengths
CoherenceLength_F = 30.0 #nm
CoherenceLength_N = 30.0 #nm

#Diffusion constants, if using dirty limit: D ~xi^2*(pi*k_B*T_c/hbar).
DN = (CoherenceLength_N/CoherenceLength_F)*(CoherenceLength_N/CoherenceLength_F) #1e-2
DF = 1E0 #1E-5

Delta = 1.55E-3 #/(k_B*T_c)

#Matsubara cutoff
nmax=1

def line(x, Gradient, Intercept):
    return Gradient*x + Intercept


def JC_model(dF, A, H, dN, T, DN, DF, Delta, nmax=1):
  
    dF = np.asarray(dF, dtype=float)
    jc = np.zeros_like(dF)

    for n in range(int(nmax)):
        omega = np.pi*k_B*T*(2*n+1)

        # diffusion wavevectors
        kN = np.sqrt(2.0*omega/DN)
        kF = np.sqrt(2.0*(omega+1j*H)/DF)

        zN = kN*dN
        zF = kF*dF

        #Stable exponential
        exp_pos = np.exp(zF)
        exp_neg = np.exp(-zF)

        sinh_zF = 0.5*(exp_pos-exp_neg)

        sinh_inv = 1.0/sinh_zF

        #Stable sech^2(x) = 1 / cosh^2(x)
        sech2 = 1.0 / np.cosh(zN)**2

        Term = ((Delta*Delta)/(omega*omega))*(1/np.cosh(zN)**2)*np.real(kF*sinh_inv)

        jc += Term
        
    if not np.all(np.isfinite(jc)):
        return np.inf * np.ones_like(jc)

    return A*np.abs(jc)



#Load the data from the file Data.txt
d,y,dy = np.loadtxt('L11 data 4.2K.txt').T 
d = d/CoherenceLength_F


Model = bmp.Curve(
    JC_model,
    d, y, dy,
    A=1.0,
    H=500,
    dN=dN_nm/CoherenceLength_N,
    T=4.2/T_c,
    DN=DN,
    DF=DF,
    Delta=Delta
)

### Limits of fitting values ###

Model.H.range(30000*0.8,30000*1.5)
Model.A.range(1E-3,10)

#Model.Gradient.dev(std=0.05, mean=None, limits=None)
#Model.Intercept.dev(std=0.5, mean=None, limits=None)

#######

#Initial values

Model.H.value = 30000
Model.A.value = 1

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()


for Htest in [100,500,1000, 30000]:
    ytest = JC_model(
        d,
        A=1,
        H=Htest,
        dN=dN_nm/CoherenceLength_N,
        T=T/T_c,
        DN=DN,
        DF=DF,
        Delta=Delta
    )
    plt.plot(d, ytest, label=f"H={Htest}")

plt.legend()
plt.show()
