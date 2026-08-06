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
H = 0.3
alpha = 0.1
SC_gap = 1.5E-3 #eV

def JC_MagneticScattering(d_F, Temperature, T_c, H, SC_gap, alpha):
    
    h = H/hbar
    Amplitude = 64*np.pi*Temperature/3
    
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff)
    
    Omega_list = (2*N_list +1)*np.pi*k_B*Temperature/hbar

    q_list = np.sqrt(2j+2*alpha+2*Omega_list/h)

    OMEGA_list = np.sqrt(Omega_list*Omega_list+SC_gap*SC_gap)

    Phi_list = SC_gap*SC_gap/((OMEGA_list+Omega_list)*(OMEGA_list+Omega_list))
    
    eta_list = np.sqrt(alpha/(alpha+1j+Omega_list/h))
     
    
    for q, phi, eta in zip(q_list,Phi_list,eta_list):
        
        Numerator = 2*q*d_F*phi*np.exp(-2*q*d_F)
        
        Denominator = np.sqrt((1-eta*eta)*phi+1)+1
        
        Term = np.real(Numerator/(Denominator*Denominator))
      
        J_c += Term
        
    return Amplitude*np.abs(J_c) #Return the current in milliamps

#Load the data from the file Data.txt
#d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

d = np.array([
    2.418, 2.438, 2.29, 2.251, 2.231, 2.123, 2.103, 2.083, 1.955, 1.936,
    1.916, 1.847, 1.827, 1.788, 1.749, 1.64, 1.621, 1.552, 1.532, 1.512,
    1.453, 1.404, 1.384, 1.158, 1.109, 1.089, 1.069, 1.05, 1.03, 1.01,
    0.961, 0.922, 0.774, 0.735, 0.715, 0.627, 0.607, 0.568, 0.459, 0.44,
    0.42, 0.479, 0.371, 0.351, 0.331, 0.272
])

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
]) 

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
])

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

#Model.Temperature.range(0.9*Temperature, 1.1*Temperature)
#Model.T_c.range(0.5*T_c,1.5*T_c)
Model.H.range(0.1*H, 10*H)
#Model.SC_gap.range(0.5*SC_gap, 1.5*SC_gap)
Model.alpha.range(0.1*alpha, 10*alpha)

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

X_axis = np.linspace(0.1, 2.5, 1000)

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
