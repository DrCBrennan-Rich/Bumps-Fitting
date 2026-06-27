# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (1) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_Ballistic.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from scipy.integrate import trapezoid


T = 5 #Temperature in K
T_c = 10 #Critical temperature in K
alpha = 1
y_max = 10*alpha

Resistance = 1.4E-3 #ohms

k_B = 1.38E-23
T = 5 #K
e = 1.6E-19

Beta = 1/(k_B*T)

def IcRn_single(d, xi, Delta):
    kB = 1#1.38e-23
    e = 1#1.6e-19
    T = 5

    #Precompute on grid
    alpha = max(d/xi, 1e-6)
    y_max = 10*alpha
    Ny = 300
    y = np.linspace(alpha, y_max, Ny)

    PhiList = np.linspace(0,2*np.pi,60)

    A = np.pi*Delta*alpha*alpha/(2*e)

    Ic = 0

    for phi in PhiList:
        Sin1 = np.sin(0.5*(phi-y))
        Sin2 = np.sin(0.5*(phi+y))

        Tanh1 = np.tanh(Delta*np.cos(0.5*(phi-y))/(2*kB*T))
        Tanh2 = np.tanh(Delta*np.cos(0.5*(phi+y))/(2*kB*T))

        Integrand = (Sin1*Tanh1+Sin2*Tanh2)/(y*y*y)

        Iphi = A*np.trapezoid(Integrand, y)

        if Iphi > Ic:
            Ic = Iphi

    return Ic

def IcRn_model(d, xi, Delta):
    return np.array([IcRn_single(di, xi, Delta) for di in np.atleast_1d(d)])


def JC_model(Thickness, CoherenceLength, SC_gap):
    
    PhiList = np.linspace(0,2*np.pi,60,endpoint=False)[:,None]
    
    AlphaList = np.maximum(Thickness/CoherenceLength, 1e-6)
    
    Ic = np.zeros_like(AlphaList)
    
    for i,Alpha in enumerate (AlphaList):
        
        yMax = max(10*Alpha,5)
        Step = max(200,int(100*yMax))
        yList = np.linspace(Alpha,yMax,Step)[None,:]
        
        SinMinus = np.sin(0.5*(PhiList-yList))
        SinPlus = np.sin(0.5*(PhiList+yList))
        
        TanhMinus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList-yList)))
        TanhPlus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList+yList)))
        
        Integrand = (1/(yList*yList*yList))*(SinMinus*TanhMinus+SinPlus*TanhPlus)
        
        Iphi = (np.pi*SC_gap*Alpha*Alpha/(2*Resistance)*trapezoid(Integrand, x=yList, axis=1))
        
        Ic[i] = np.max(np.abs(Iphi))
        
    return Ic


def JC_model2(Thickness, CoherenceLength, SC_gap):

    PhiList = np.linspace(0,2*np.pi,100)

    AlphaList = np.maximum(Thickness/CoherenceLength,1e-6)

    Ic = np.zeros_like(AlphaList)

    prefactor = np.pi*SC_gap/(2*Resistance)

    for i, Alpha in enumerate(AlphaList):

        def Integrand(y):

            SinMinus = np.sin(0.5*(PhiList-y))
            SinPlus  = np.sin(0.5*(PhiList+y))

            TanhMinus = np.tanh(
                0.5*Beta*SC_gap*np.cos(0.5*(PhiList-y))
            )

            TanhPlus = np.tanh(
                0.5*Beta*SC_gap*np.cos(0.5*(PhiList+y))
            )

            return (SinMinus*TanhMinus + SinPlus*TanhPlus)/y**3

        Iphi, err = quad_vec(
            Integrand,
            Alpha,
            np.inf,
            epsrel=1e-8,
            epsabs=1e-8
        )

        Iphi *= prefactor*Alpha**2

        Ic[i] = np.max(np.abs(Iphi))

    return Ic
        

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('L11 data 4.2K.txt').T
#y = np.log(y)
Model = bmp.Curve(JC_model, d, y, dy)

### Limits of fitting values ###

Model.SC_gap.range(1,50) 
Model.CoherenceLength.range(0.001,100)  

#Model.Gradient.dev(std=0.05, mean=None, limits=None)
#Model.Intercept.dev(std=0.5, mean=None, limits=None)

#Initial values

Model.SC_gap.value = 31 #eV
Model.CoherenceLength.value = 0.2

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Plot the datapoints used in the fitting
plt.errorbar(
    d, y, yerr=dy,
    fmt='o',
    capsize=3,
    label='Experimental data'
)


for CoherenceLength_test in [0.4]:
    ytest = JC_model(
        d,
        CoherenceLength=CoherenceLength_test,
        SC_gap = 4.5E-3
    )
    plt.plot(d, ytest, label=f"Coherence Length={CoherenceLength_test}", color='pink')
    plt.yscale('log')

plt.legend()
plt.savefig("Ballistic.svg", format="svg")
plt.show()


for SC_gap_test in [1.5E-3,4E-3,8E-3,12E-3]:
    ytest = JC_model(
        d,
        CoherenceLength=8E-2,
        SC_gap = SC_gap_test
    )
    plt.plot(d, ytest, label=f"Gap={SC_gap_test}")

plt.legend()
plt.show()
