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
alpha = 0.01
SC_gap = 1.5E-3 #eV

def JC_MagneticScattering(d_F, Temperature, T_c, H, SC_gap, alpha):
    
    h = H/hbar
    Amplitude = 64*np.pi*Temperature/3
    
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff)
    #
    Omega_list = (2*N_list +1)*np.pi*k_B*Temperature

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
d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

d = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0, 0.30000000000000004, 0.44999999999999996, 0.6000000000000001, 0.75, 0.8999999999999999, 1.0499999999999998, 1.2000000000000002, 1.35, 0.15000000000000002])#, 1.5, 1.6500000000000001, 1.7999999999999998])
y = np.array([np.float64(25.485540677019983), np.float64(20.24388234661639), np.float64(9.229536446202395), np.float64(2.3917306615253358), np.float64(5.332381933490123), np.float64(6.262335355328344), np.float64(5.555634474421091), np.float64(7.14605149171436), np.float64(4.670040152958881), np.float64(4.466469736108299), np.float64(4.402734073344201), np.float64(2.8669437006283083), np.float64(2.198601946982659), np.float64(0.31666206152922616), np.float64(1.018208817810008), np.float64(6.094153299948496), np.float64(6.307979995277196), np.float64(5.485744923060322), np.float64(2.581500267199711), np.float64(0.3440655265061016), np.float64(1.3730251946938403), np.float64(0.951189258094282), np.float64(0.08461134762633567), np.float64(38.13333333333333)])#, np.float64(0.1905), np.float64(0.3), np.float64(0.12166666666666666)])
dy = np.array([np.float64(1.1541353059558421), np.float64(1.3282852847429805), np.float64(0.43555339620836153), np.float64(0.183136722622837), np.float64(0.32397334928648797), np.float64(0.16637832288545412), np.float64(0.4313857996461637), np.float64(0.2067104216127845), np.float64(0.26935663136737165), np.float64(0.11061774240230109), np.float64(0.5189157313556868), np.float64(0.13401844281653402), np.float64(0.1469797151953465), np.float64(0.051007687876081016), np.float64(0.08432750457227471), np.float64(0.12104294939270999), np.float64(0.2824216946228165), np.float64(0.19224653742922326), np.float64(0.03978367734086671), np.float64(0.018540196895687144), np.float64(0.20333691412042817), 0.0053777623606668535, 0.004048255991005446, np.float64(1.7975291683617007)])#, np.float64(0.0035000000000000027), np.float64(0.04999999999999999), np.float64(0.010137937550497038)])

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
