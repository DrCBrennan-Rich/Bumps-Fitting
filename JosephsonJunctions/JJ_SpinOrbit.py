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

d = np.array([
    2.418, 2.438, 2.29, 2.251, 2.231, 2.123, 2.103, 2.083, 1.955, 1.936,
    1.916, 1.847, 1.827, 1.788, 1.749, 1.64, 1.621, 1.552, 1.532, 1.512,
    1.453, 1.404, 1.384, 1.158, 1.109, 1.089, 1.069, 1.05, 1.03, 1.01,
    0.961, 0.922, 0.774, 0.735, 0.715, 0.627, 0.607, 0.568, 0.459, 0.44,
    0.42, 0.479, 0.371, 0.351, 0.331, 0.272
]) #

y = np.array([
    0.00699425, 0.0078925, 0.013320000000000002, 0.014122499999999998,
    0.025375, 0.10813610571578758, 0.12869791639745418, 0.12872074116519092,
    0.1576875, 0.19116130513139204, 0.2794563049095635, 0.16480457130267517,
    0.22519604251290912, 0.2147201206749916, 0.29277927044068863,
    0.10372115678110677, 0.10148694224915314, 0.015675129846466566,
    0.05681675965155805, 0.04702147096391103, 0.1311, 0.575868126049855,
    0.5854827027222074, 1.5882281427913232, 2.0508648451755738,
    3.4725234237693314, 4.093075134553839, 4.114645659311062,
    4.01905617358874, 4.636714834438517, 8.06676028826663, 9.37677621440921,
    15.296359148827252, 16.7295, 18.21934959116588, 14.409216666666666,
    14.910333333333334, 9.78056007112751, 1.8367327942466058,
    5.964761752065373, 6.514800423728491, 1.8710824247921065,
    13.332891880340954, 13.9206875, 14.946451056265937,
    34.849999999999994
]) #

dy = np.array([
    0.009696086439899347, 0.009132366889257133, 0.002327573844156186,
    0.0010838011810290656, 0.005411330705103877, 0.0018541940391505435,
    0.0012732532397040698, 0.0010827887503948638, 0.0029892515785728065,
    0.022509418754522274, 0.013406533112136592, 0.004314025115285019,
    0.004232886017248341, 0.020450251732958894, 0.025948621075078376,
    0.00621581930372137, 0.006945663211863579, 0.0011761830555589625,
    0.0010460182649376032, 0.002794379735734216, 0.010121264743104002,
    0.003535997110096228, 0.006370438899523077, 0.021277803185625884,
    0.034404666730416006, 0.008178965051041997, 0.014781311592187553,
    0.021628692560245775, 0.02348659112691888, 0.016374045211460568,
    0.05387639673842677, 0.12203944272943992, 0.31898692745103996,
    0.3704999999999998, 0.79934959116588, 1.199686673098902,
    0.2896666666666663, 0.4354436646959341, 0.4689634154068645,
    0.02114035317899629, 0.5337258517802882, 0.16824712452052695,
    2.7333575623258035, 5.937062499999999, 2.047282020666608,
    3.703501721344273
]) #

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
