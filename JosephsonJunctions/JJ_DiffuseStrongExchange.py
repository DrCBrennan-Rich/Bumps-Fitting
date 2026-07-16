# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (18) from https://doi.org/10.1088/1367-2630/17/11/113022
#Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_DiffuseStrongExchange.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

k_B = 8.617333262E-5 #eV/K
SC_gap = 1.5E-3 #eV
Temperature = 4.2 #K

T_c = 9.2
Resistivity = 16.8 #ohm nm
hbar = 6.582E-16 #eV*s
FermiVelocity = 3.3E5*1E9 #nm/s
MeanFreePath = 0.283496 #nm
DiffusionCoeff = FermiVelocity*MeanFreePath/3 #nm^2/s
CoherenceLength = np.sqrt(DiffusionCoeff*hbar/(2*np.pi*k_B*T_c))
AR = 5.7*1E3 #Ohm nm^2
FreqCutoff=5
Resistivity_N = 87 #Ohm nm

d_N = 0.4
xi_N = 5
gamma_BSN = 1
gamma_NF = 0.01

SpinScatterTime = np.inf
gamma_BNF = 0.001
gamma_BSF = 1
Area = np.pi*(1.5E3)*(1.5E3) #Area of the gate in nm^2

def Trancendental_Quartic(Chi_vec,gamma,Omega,eta,theta):
    #Equation 20 and 22
    
    Chi = Chi_vec[0]+1j*Chi_vec[1]
    S = np.sin(theta)
    u = np.sqrt(Omega+eta*(1-Chi*Chi))
    Residual = Chi**4+(2*gamma*u*S)*Chi**3+((gamma*u)**2-1)*Chi**2-(gamma*u*S)*Chi+0.25*S*S
    return [np.real(Residual), np.imag(Residual)]

def Solve_Quartic_Exact(gamma,Omega,theta):
    #Solve equation 20 or 22 if eta = 0
    
    S = np.sin(theta)
    u = np.sqrt(Omega)

    coeffs = [1,2*gamma*u*S,(gamma*u)**2-1,-(gamma*u*S),0.25*S*S]

    Roots = np.roots(coeffs)
    return Roots

def Pick_Root(Roots,gamma,Omega,theta):
    #Four roots exist, selects the correct one using equation 19 or 21
    
    LHS = 2*gamma*np.sqrt(Omega)*Roots
    RHS = np.sin(theta-2*np.arcsin(Roots))
    
    i = np.argmin(np.abs(RHS-LHS))
    return Roots[i]

def Find_Theta_NF(d_N, Omega, xi_N, theta_NS, gamma_BSN, theta_S):
    #Equation A5
    Difference = theta_NS-theta_S
    
    Term1 = (np.real(Omega)*d_N*d_N)*np.sin(theta_NS)/(2*xi_N*xi_N)
    Term2 = (d_N*np.sin(Difference))/(gamma_BSN*xi_N)
    theta_NF = Term1 + Term2 + theta_NS
    
    return theta_NF

def Find_Theta_NS_Initial(d_N, Omega, xi_N, gamma_BSN, theta_S):
    #If eta and gamma_NF = 0 then this function will find theta_NS from equation A8
    A = (np.real(Omega)*d_N*gamma_BSN)/(xi_N*np.sin(theta_S))
    B = (A+(1/np.tan(theta_S)))*(A+(1/np.tan(theta_S)))
    C = np.sqrt(1/(B+1))
    
    theta_NS = np.arcsin(C)   
    
    return theta_NS

def Find_Theta_NS_Initial2(d_N, Omega, xi_N, gamma_BSN, theta_S):
    #If eta and gamma_NF = 0 then this function will find theta_NS, based off equation A10/11
    A = gamma_BSN*np.real(Omega)*d_N/xi_N
    Lambda = np.sqrt(1+2*A*np.cos(theta_S)+A*A)
    
    theta_NS = np.arcsin(np.sin(theta_S)/Lambda)   
    
    return theta_NS

def All_Equations(ChiAndAngles, Omega, eta, gamma_BNF, gamma_NF, gamma_BSN,
           d_N, xi_N, theta_S):

    ChiReal = ChiAndAngles[0]
    ChiImaginary = ChiAndAngles[1]
    Chi = ChiReal + 1j*ChiImaginary
    
    theta_NS_Real = ChiAndAngles[2]
    theta_NS_Imaginary = ChiAndAngles[3]
    theta_NS = theta_NS_Real + 1j*theta_NS_Imaginary
    
    theta_NF_Real = ChiAndAngles[4]
    theta_NF_Imaginary = ChiAndAngles[5]
    theta_NF = theta_NF_Real + 1j*theta_NF_Imaginary
    
    S = np.sin(theta_NF)
    u = np.sqrt(Omega + eta*(1-Chi*Chi))
    Difference = theta_NS - theta_S

    #Equation 22, complex
    eq22= (Chi*Chi*Chi*Chi
        + (2*gamma_BNF*u*S)*Chi*Chi*Chi
        + ((gamma_BNF*u)*(gamma_BNF*u)-1)*Chi*Chi
        - (gamma_BNF*u*S)*Chi
        + 0.25*S*S)
    
    #Equation A5
    eqA5 = theta_NF - (
        (np.real(Omega)*d_N*d_N*np.sin(theta_NS))/(2*xi_N*xi_N)
        + (d_N*np.sin(Difference))/(gamma_BSN*xi_N)
        + theta_NS)
    
    #Equation A8
    eqA8 = (-2*gamma_NF*gamma_BSN*np.sqrt(Omega + eta*(1-Chi**2))*Chi
        - (np.real(Omega)*d_N*gamma_BSN/xi_N)*np.sin(theta_NS)
        - np.sin(Difference))
    
    eq22_real = np.real(eq22)
    eq22_imaginary = np.imag(eq22)
   
    eqA5_real = np.real(eqA5)
    eqA5_imaginary = np.imag(eqA5)
    
    eqA8_real = np.real(eqA8)
    eqA8_imaginary = np.imag(eqA8)

    return [eq22_real, eq22_imaginary, 
            eqA5_real, eqA5_imaginary,
            eqA8_real, eqA8_imaginary]


def JC_DiffuseExchange(d_F, Temperature, Resistivity_N, SpinScatterTime, 
                       CoherenceLength, H, gamma_NF, gamma_BSN, d_N, xi_N, 
                       SC_gap, Area):
    
    Resistivity_F = (Resistivity_N*xi_N)/(gamma_NF*CoherenceLength)
    Amplitude = Area*(16*np.pi*k_B*Temperature)/(Resistivity_F)
    
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff)
    #"Omega" in this work will refer to Omega-tilda in the original paper
    Omega_list = (Temperature/T_c)*(2*N_list+1)+(H/(np.pi*k_B*T_c))*1j

    eta = hbar/(np.pi*SpinScatterTime*k_B*T_c)
    gamma_BNF = 0.001#AR/(CoherenceLength*Resistivity) Defined as this value in the paper, needs to be fitted
    gamma_list = np.sqrt(Omega_list+eta)/CoherenceLength
    
    for gamma, w in zip(gamma_list, Omega_list):
        #Define theta_S from equation 5
        theta_S = np.arctan(SC_gap/(np.pi*k_B*T_c*np.real(w)))
        #Find the intial angles taking gamma_NF and eta = 0
        theta_NS_initial = Find_Theta_NS_Initial(d_N, w, xi_N, gamma_BSN, theta_S)
        theta_NF_initial = Find_Theta_NF(d_N, w, xi_N, theta_NS_initial, gamma_BSN, theta_S)
        
        #Exact solution of the quartic equation 20/22 and then selecting the real root
        
        #####  For solving SNF boundary  #######
        Roots = Solve_Quartic_Exact(gamma_BNF, w, theta_NF_initial)
        Chi_initial = Pick_Root(Roots, gamma_BNF, w, theta_NF_initial)     
        
        Guess = [np.real(Chi_initial), np.imag(Chi_initial),
                 np.real(theta_NS_initial), np.imag(theta_NS_initial), 
                 np.real(theta_NF_initial), np.imag(theta_NF_initial)]
              
        gamma_NF_Steps = np.linspace(0,gamma_NF,10)
        EtaSteps = np.linspace(0,eta,10)
        
        for gammaIntermediate in gamma_NF_Steps:
            #Relax the gamma_NF = 0 condition
            Solution = fsolve(All_Equations,
                Guess, args=(w, 0, gamma_BNF, gammaIntermediate, gamma_BSN,
                      d_N, xi_N, theta_S))
            Guess = [Solution[0], Solution[1], 
                     Solution[2], Solution[3],
                     Solution[4], Solution[5]]
        
        for EtaIntermediate in EtaSteps:
            #Relax eta=0 condition
            Solution = fsolve(All_Equations,
                Guess,
                args=(w, EtaIntermediate, gamma_BNF, gamma_NF, gamma_BSN,
                      d_N, xi_N, theta_S))
            Guess = [Solution[0], Solution[1], 
                     Solution[2], Solution[3],
                     Solution[4], Solution[5]]
        
        Chi1 = Solution[0] + 1j*Solution[1]
        
        Roots2 = Solve_Quartic_Exact(gamma_BSF, w, theta_S)
        
        Chi2_Initial = Pick_Root(Roots2, gamma_BSF, w, theta_S)
        
        EtaSteps = np.linspace(0,eta,10)
        Guess = [Chi2_Initial.real, Chi2_Initial.imag]
        
        for EtaIntermediate in EtaSteps:
            #Relax eta=0 condition
            Solution = fsolve(
                Trancendental_Quartic,
                Guess,
                args=(gamma_BSF, w, EtaIntermediate, theta_S)
            )
            Guess = [Solution[0], Solution[1]]
         
        Chi2 = Solution[0] + 1j*Solution[1]
        
        Term = np.real(gamma*np.exp(-gamma*d_F)*Chi1*Chi2)
        J_c += Term
           
    return Amplitude*np.abs(J_c)

#Load the data from the file Data.txt
d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

OrderingIndex = np.argsort(d)
d = d[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    Temperature=Temperature,
    Resistivity_N = 87,#Ohm nm
    gamma_NF=gamma_NF,
    gamma_BSN=gamma_BSN,
    d_N=d_N,
    xi_N=xi_N,
    SC_gap=SC_gap,
    CoherenceLength=CoherenceLength,
    Area = Area)

### Limits of fitting values ###

#Model.CoherenceLength.range(1E-3,10)
#Model.H.range(1E-5,3E-3)
#Model.Temperature.range(1,10)
Model.SpinScatterTime.range(1E-18,5E-11)
Model.gamma_NF.range(1E-4,10)
Model.Area.range(5E6,1E16)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values

Model.CoherenceLength.value = 2.087 #nm
Model.H.value = 0.679
Model.Temperature.value = 4.2
Model.SpinScatterTime.value = 100E-15
Model.Resistivity_N.value = 87 #Ohm nm
Model.gamma_NF.value = 1
Model.SC_gap.value = 1.5E-3 #eV
Model.xi_N.value = 30 #nm
Model.d_N.value = 7.5 #nm
Model.gamma_BSN.value = 1.92
Model.Area.value = 7068583.470577034

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


X_axis = np.linspace(0.1, 2, 1000)
J_0 = Area*np.pi*k_B*T_c/(Resistivity*CoherenceLength)

# for gamma_NF_test in [0.01, 0.1, 1]:

#     plt.figure(figsize=(8,6))

#     for dN_test in [5,10]:

#         ytest = JC_DiffuseExchange(
#             X_axis,
#             Temperature=T_c/2,
#             Resistivity_N=Resistivity_N,
#             CoherenceLength=CoherenceLength,
#             SpinScatterTime=SpinScatterTime,
#             H=0.6*h*np.pi*k_B*T_c,
#             gamma_NF=gamma_NF_test,
#             gamma_BSN=0.001/gamma_NF_test,
#             d_N=dN_test,
#             xi_N=30, SC_gap = SC_gap
#         )

#         plt.plot(
#             X_axis / CoherenceLength,
#             ytest / J_0,
#             linewidth=3,
#             label=rf"$d_N={dN_test}\,\mathrm{{nm}}$"
#         )

#     plt.title(rf"$\gamma_{{NF}}={gamma_NF_test}$", fontsize=20)
#     plt.xlabel(r"$d_F/\xi_F$", fontsize=16)
#     plt.ylabel(r"$J_c/J_0$", fontsize=16)
#     plt.yscale("log")
#     plt.legend(fontsize=14)
#     plt.tight_layout()
#     plt.show()
# #plt.savefig("Changing_gamma_NF.svg", format="svg")
# #plt.show()

for test in [0.001]:
    ytest = JC_DiffuseExchange(
        X_axis,
        Temperature=4.2,
        Resistivity_N= 87,#ohm nm,
        CoherenceLength=2.087,
        SpinScatterTime=1E25,
        H=0.679,
        gamma_NF=test,
        gamma_BSN=1.92,
        d_N=7.5,
        xi_N=30,
        SC_gap = 1.5E-3, #eV
        Area = Area
    )
    plt.plot(X_axis, 50000000*ytest, label=f"TestVariable={test}", linewidth=3,)
plt.yscale('log')
plt.legend(fontsize=24)
plt.show()
