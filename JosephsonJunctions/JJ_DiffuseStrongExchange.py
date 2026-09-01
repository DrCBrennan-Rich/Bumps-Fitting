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
from scipy import constants

#As a general rule of thumb throughout this script energies will be in 
#electron-volts, lengths will be in nm, and electrical parameters will be in
#their SI units (Volts, Ampere, Ohm, Coulomb).

k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K
hbar = constants.physical_constants['reduced Planck constant in eV s'][0] #eV*s

FreqCutoff=20
StepNumber = 10
T_c = 9.2

Temperature=4.2 #K
JunctionResistance = 1.55E-3 #Ohms
InterfaceResistance = 5700 #Ohm nm^2
Area = np.pi*(1.5E3)*(1.5E3) #Area of the gate in nm^2

gamma_BNF = 1000
gamma_BSF = 1

Amplitude = 250#90892.9#827795
H=0.679 #eV
CoherenceLength= 1.99 #nm
gamma_BSN = 0.398271
SC_gap = 1.5E-3 #eV
d_N1 = 5 #Thickness of the left hand normal metal nm
d_N2 = 10 #Thickness of the right hand normal metal nm
xi_N = 30
gamma_NF = 7.98175e-08
Resistivity_F =  26404.2 #ohm nm
Resistivity_N = 87 #ohm nm
eta = 0
DeadLayer = -0.410589 

#Green function: F = exp(j*chi)*sin(theta)

def Trancendental_Quartic(Chi_vec,gamma,Omega,eta,theta):
    """Define the residual (f(x) = 0) for the trancendental quartic.

    Defines the residual for the quaritc equation 20 (or 22) in
    order for it to be supplied to Fsolve. To do this, the real and imaginary
    residuals have to calculated seperately and returned as a 2x1 array.

    Args:
        Chi_vec (numpy.ndarray): Two long array containing the real and 
            imaginary components of Chi (unitless).
        gamma (float): Suppression parameter at the boundary.
        Omega (complex): Dimensionless Matsurbara frequency (unitless).
        eta (float): Spin-flip scattering parameter, defined as: eta = hbar/(pi*k_B*T_c*tau_m) 
            where tau_m is the spin-flip scattering time (unitless).
        theta (float): Pairing angle (radians).

    Returns:
        Residual (numpy.ndarray): Array containing the real and complex 
        components of the residual.

    Notes:
    """
    #Equation 20 and 22
    
    Chi = Chi_vec[0]+1j*Chi_vec[1]
    S = np.sin(theta)
    u = np.sqrt(Omega+eta*(1-Chi*Chi))
    Residual = Chi**4+(2*gamma*u*S)*Chi**3+((gamma*u)**2-1)*Chi**2-(gamma*u*S)*Chi+0.25*S*S
    return [np.real(Residual), np.imag(Residual)]

def Solve_Quartic_Exact(gamma,Omega,theta):
    """Solve a quartic equation for the exact roots.

    Solves a quartic equation corresponding to Eq. 20 (or Eq. 22) 
    when eta = 0 using NumPy's polynomial root solver.

    Args:
        gamma (float): Dimensionless parameter controlling the strength
            of the quartic terms (unitless).
        Omega (complex): Dimensionless Matsurbara frequency (unitless).
        theta (float): Pairing angle (radians).

    Returns:
        Roots (numpy.ndarray): Array containing the four (possibly complex)
            roots of the quartic equation.

    Notes:
        The polynomial solved is

            x^4 + 2*gamma*sqrt(Omega)*sin(theta)*x^3
            + (gamma^2*Omega - 1)*x^2
            - gamma*sqrt(Omega)*sin(theta)*x
            + sin(theta)^2/4 = 0.

        The roots are calculated using ``numpy.roots`` and are not
        guaranteed to be returned in any particular order.
    """
    
    S = np.sin(theta)
    u = np.sqrt(Omega)

    coeffs = [1,2*gamma*u*S,(gamma*u)**2-1,-(gamma*u*S),0.25*S*S]

    Roots = np.roots(coeffs)
    return Roots

def Pick_Root(Roots,gamma,Omega,theta):
    """Select the correct root of the quartic equation.

    Calculates the correct root from the four returned from the quartic by 
    using Eq. 19 (or Eq. 21). The left hand side (LHS) and right hand side (RHS)
    are calculated and compared with the value closest to 0 returned.

    Args:
        Roots (numpy.ndarray): List of (complex) roots (unitless).
        gamma (float): Dimensionless parameter controlling the strength
            of the quartic terms (unitless).
        Omega (complex):  Dimensionless Matsurbara frequency (unitless).
        theta (float): Pairing angle (radians).

    Returns:
        Root (complex): Physically correct root (unitless).

    Notes:
        The equation being solved is

           2*gamma*sqrt[Omega]*Sin(theta/2) = Sin(theta_S-theta)
           where we then define Root = Sin(theta/2).
    """
    
    LHS = 2*gamma*np.sqrt(Omega)*Roots
    RHS = np.sin(theta-2*np.arcsin(Roots))
    
    #Find the index of the position where RHS most closely matches LHS
    i = np.argmin(np.abs(RHS-LHS))
    return Roots[i]

def Find_Theta_NF(d_N, Omega, xi_N, theta_NS, gamma_BSN, theta_S):
    """Calculate the suppresion parameter between the normal and ferromagnetic
    boundary: theta_NF.

    Calculates theta_NF from the known theta_NS and theta_S values according to
    Eq. A5.

    Args:
        d_N (float): Thicknesses of the normal metal (nm).
        Omega (float): Dimensionless Matsurbara frequency (real component, unitless).
        xi_N (float): Coherence length in the normal metal (nm).
        theta_NS (float): Pairing angle (radians).
        gamma_BSN (float): Boundary suppresion parameter between superconductor
            and normal metal (unitless).
        theta_S (float): Superconducting pairing angle (radians).

    Returns:
        theta_NF (float): Pairing angle between normal and ferromagnetic 
            materials (radians).

    Notes:
        The equation being solved is:

           theta_NF = Omega*d_N^2*Sin(theta_NS)/(2*xi_N^2) + 
                       d_N*Sin(theta_NS-theta_S)/(gamma_BSN*xi_N) + theta_NS
    """
    Difference = theta_NS-theta_S
    
    Term1 = (np.real(Omega)*d_N*d_N)*np.sin(theta_NS)/(2*xi_N*xi_N)
    Term2 = (d_N*np.sin(Difference))/(gamma_BSN*xi_N)
    theta_NF = Term1 + Term2 + theta_NS
    
    return theta_NF

def Find_Theta_NS_Initial(d_N, Omega, xi_N, gamma_BSN, theta_S):
    """Calculate the suppresion parameter between the normal and 
    superconducting boundary: theta_NS, for eta, gamma_NF = 0

    Calculates the initial theta_NS value for the exactly solvable situation 
    where the scattering parameter and normal-ferromagnetic suppresion 
    parameter (eta and gamma_NF) are both 0, using equation A8. This value can
    then be used as a starting point to solve the trancendental equation for 
    eta, gamma_NF =/= 0.

    Args:
        d_N (float): Thicknesses of the normal metal (nm).
        Omega (complex): Dimensionless Matsurbara frequency (unitless).
        xi_N (float): Coherence length in the normal metal (nm).
        theta_NS (float): Pairing angle (radians).
        gamma_BSN (float): Boundary suppresion parameter between superconductor
            and normal metal (unitless).
        theta_S (float): Superconducting pairing angle (radians).

    Returns:
        theta_NS (float): Pairing angle between normal and superconducting 
            materials (radians).

    Notes:
        The equation being solved is:
            
           0 = Real[Omega]*d_N*gamma_BSN*Sin(theta_NS) + Sin(theta_NS-theta_S)
    """

    A = (np.real(Omega)*d_N*gamma_BSN)/(xi_N*np.sin(theta_S))
    B = (A+(1/np.tan(theta_S)))*(A+(1/np.tan(theta_S)))
    C = np.sqrt(1/(B+1))
    
    theta_NS = np.arcsin(C)   
    
    return theta_NS

def Find_Theta_NS_Initial2(d_N, Omega, xi_N, gamma_BSN, theta_S):
    """Calculate the suppresion parameter between the normal and 
    superconducting boundary: theta_NS, for eta, gamma_NF = 0

    Calculates the initial theta_NS value for the exactly solvable situation 
    where the scattering parameter and normal-ferromagnetic suppresion 
    parameter (eta and gamma_NF) are both 0, using equation A10 and A11. This 
    value can then be used as a starting point to solve the trancendental 
    equation for eta, gamma_NF =/= 0.

    Args:
        d_N (float): Thicknesses of the normal metal (nm).
        Omega (complex): Dimensionless Matsurbara frequency (unitless).
        xi_N (float): Coherence length in the normal metal (nm).
        theta_NS (float): Pairing angle (radians).
        gamma_BSN (float): Boundary suppresion parameter between superconductor
            and normal metal (unitless).
        theta_S (float): Superconducting pairing angle (unitless).

    Returns:
        theta_NS (float): Pairing angle between normal and superconducting 
            materials (radians).

    Notes:
        The equations being solved are:
            
            Sin(theta_NS) = lambda*Sin(theta_S)
            
            1/(lambda^2) = 1 + 2*Cos(theta_S)*gamma_BSN*Real[Omega]*d_N/xi_N
                            + gamma_BSN^2*Real[Omega]^2*d_N^2/xi_N^2
    """

    A = gamma_BSN*np.real(Omega)*d_N/xi_N
    Lambda = np.sqrt(1+2*A*np.cos(theta_S)+A*A)
    
    theta_NS = np.arcsin(np.sin(theta_S)/Lambda)   
    
    return theta_NS

def All_Equations(ChiAndAngles, Omega, eta, gamma_BNF, gamma_NF, gamma_BSN,
           d_N, xi_N, theta_S):
    
    """Form the residuals for the real and complex components of the three
    simultaneous equations.
    
    Calculates the residuals (=0 functions) of the three equations: A5, A8 and
    Eq. 22 which can then be solved simultanously using fsolve. Since Chi,
    theta_NS, and theta_NF in principle can all be complex, six (three real
    and three imaginary) are calculated.

    Args:
        ChiAndAngles (list): A six element list where the elements must be:
            [0] - Real component of the interface constant, Chi (unitless).
            [1] - Imaginary component of the interface constant, Chi (unitless).
            [2] - Real component of the pairing angle on NS boundary (radians).
            [3] - Imaginary component of the pairing angle on NS boundary (radians).
            [4] - Real component of the pairing angle on NF boundary (radians).
            [5] - Imaginary component of the pairing angle on NF boundary (radians).
        Omega (complex): Dimensionless Matsurbara frequency (unitless).
        eta (float): Spin-flip scattering parameter, defined as: eta = hbar/(pi*k_B*T_c*tau_m) 
            where tau_m is the spin-flip scattering time (unitless).
        gamma_BNF (float): Boundary suppresion parameter between the normal metal
            and the ferromagnet (unitless).
        gamma_NF (float): Suppresion parameter between the normal metal
            and the superconductor (unitless).
        gamma_BSN (float): Boundary suppresion parameter between superconductor
            and normal metal (unitless).
        d_N (float): Thickness of the normal metal (nm).
        xi_N (float): Coherence length in the normal metal (nm).
        theta_S (float): Superconducting pairing angle (unitless).
        

    Returns:
        AllEquations (list): A six element list where the elements will be:
            [0] - Real residual of equation 22 (unitless).
            [1] - Imaginary residual of equation 22 (unitless).
            [2] - Real residual of equation A5 (unitless).
            [3] - Imaginary residual of equation A5 (unitless).
            [4] - Real residual of equation A8 (unitless).
            [5] - Imaginary residual of equation A8 (unitless).

    Notes:
        The equations being solved are:
            
            Eq22: 0 = Chi^4 + 2*gamma_BNF*u*S*Chi^3 + ((gamma_BNF*u)^2-1)*Chi^2 
                - gamma_BNF*u*S*Chi + S*S/4
                
            EqA5: 0 = theta_NF - (Real[Omega]*d_N^2*Sin(theta_NS))/(2*xi_N^2)
                + (d_N*np.sin(Difference))/(gamma_BSN*xi_N) + theta_NS
                
            EqA8: 0 = -2*gamma_NF*gamma_BSN*u*Chi - Sin(Difference)
                - (Real[Omega]*d_N*gamma_BSN/xi_N)*Sin(theta_NS)
            
            Where:
                Difference = theta_NS - theta_S
                S = Sin(theta_NF)
                u = sqrt[Omega + eta*(1-Chi^2)]
    """
    
    #Seperate the real and imaginary equations
    ChiReal = ChiAndAngles[0]
    ChiImaginary = ChiAndAngles[1]
    Chi = ChiReal + 1j*ChiImaginary
    
    theta_NS_Real = ChiAndAngles[2]
    theta_NS_Imaginary = ChiAndAngles[3]
    theta_NS = theta_NS_Real + 1j*theta_NS_Imaginary
    
    theta_NF_Real = ChiAndAngles[4]
    theta_NF_Imaginary = ChiAndAngles[5]
    theta_NF = theta_NF_Real + 1j*theta_NF_Imaginary
    
    Chi2 = Chi*Chi
    Chi3 = Chi*Chi*Chi
    Chi4 = Chi*Chi*Chi*Chi
    
    S = np.sin(theta_NF)
    u = np.sqrt(Omega + eta*(1-Chi2))
    Difference = theta_NS - theta_S

    #Equation 22, complex
    eq22= (Chi4
        + (2*gamma_BNF*u*S)*Chi3
        + ((gamma_BNF*u)*(gamma_BNF*u)-1)*Chi2
        - (gamma_BNF*u*S)*Chi
        + 0.25*S*S)
    
    #Equation A5
    eqA5 = theta_NF - (
        (np.real(Omega)*d_N*d_N*np.sin(theta_NS))/(2*xi_N*xi_N)
        + (d_N*np.sin(Difference))/(gamma_BSN*xi_N)
        + theta_NS)
    
    #Equation A8
    eqA8 = (-2*gamma_NF*gamma_BSN*u*Chi
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

def Find_SF_Boundary_Chi(gamma_BSF, Omega, theta_S, eta, StepNumber):
    """Calculate the boundary constant, Chi, between the superconducting and 
    ferromagnet interface.

    This function summons the required functions to first solve the boundary
    constant for the exact solution when eta = 0; then selects the
    correct root; and finally performs the stepping operation to progress from
    the eta = 0 case to the desired eta. Fsolve is called at each step using
    the previous solution as the initial search point for the next solution.

    Args:
        gamma_BSF (float): Boundary suppresion parameter between superconductor
            and ferromagnet (unitless).
        Omega (complex): Dimensionless Matsurbara frequency (unitless).
        theta_S (float): Superconducting pairing angle (unitless).
        eta (float): Spin-flip scattering parameter, defined as: eta = hbar/(pi*k_B*T_c*tau_m) 
            where tau_m is the spin-flip scattering time (unitless).
        StepNumber (int): Number of steps performed between 0 and eta.
        

    Returns:
        Chi_SF (complex): Boundary constant between the superconducting and 
            ferromagnet interface (unitless).

    Notes:

    """
    
    Roots = Solve_Quartic_Exact(gamma_BSF, Omega, theta_S)
    
    Chi2_Initial = Pick_Root(Roots, gamma_BSF, Omega, theta_S)
    
    EtaSteps = np.linspace(0,eta,StepNumber)
    Guess = [Chi2_Initial.real, Chi2_Initial.imag]
    
    for EtaIntermediate in EtaSteps:
        #Relax eta=0 condition
        Solution, Info, ErrorCheck, Message= fsolve(
            Trancendental_Quartic,
            Guess,
            args=(gamma_BSF, Omega, EtaIntermediate, theta_S),
            full_output=True)
        
        if ErrorCheck == 0:
            raise RuntimeError(
                f"fsolve failed: {Message}\n"
                f"Omega={Omega}, eta={EtaIntermediate}, "
                f"gamma_NF={gamma_NF}\n"
                f"Guess={Guess}\n"
                f"Solution={Solution}")
            
        Guess = [Solution[0], Solution[1]]
     
    Chi_SF = Solution[0] + 1j*Solution[1]
    
    return Chi_SF

def Find_SNF_Boundary_Chi(gamma_BNF, Omega, theta_NF_initial, theta_NS_initial, 
                          eta, theta_S, gamma_NF, StepNumber):
    """Calculate the boundary constant, Chi, between the superconducting/normal
    and ferromagnet interface.

    This function summons the required functions to first solve the boundary
    constant for the exact solution when gamma_NF, eta = 0; then selects the
    correct root; and finally performs the stepping operation to progress from
    the gamma_NF = 0 case to the desired gamma_NF; followed by stepping from
    eta = 0 to the desired eta. Fsolve is called at each step using the 
    previous solution as the initial search point for the next solution.

    Args:
        gamma_BNF (float): Boundary suppresion parameter between the normal 
        metal and ferromagnet (unitless).
        
        Omega (complex): Dimensionless Matsurbara frequency (unitless).
        theta_NF_initial (float): Pairing angle between normal and 
            ferromagnetic materials (radians). 
        theta_NS_initial (float): Pairing angle between normal and 
            superconducting materials (radians). 
        eta (float): Spin-flip scattering parameter, defined as: eta = hbar/(pi*k_B*T_c*tau_m) 
            where tau_m is the spin-flip scattering time (unitless).
        theta_S (float): Superconducting pairing angle (unitless).
        gamma_NF (float): Suppresion parameter between the normal metal
            and the superconductor (unitless).
        StepNumber (int): Number of steps performed between 0 and eta.
        

    Returns:
        Chi_SNF (complex): Boundary constant between the superconducting/normal
            /metal and ferromagnet interface (unitless).

    Notes:

    """
    Roots = Solve_Quartic_Exact(gamma_BNF, Omega, theta_NF_initial)
    
    Chi_initial = Pick_Root(Roots, gamma_BNF, Omega, theta_NF_initial)     
    
    Guess = [np.real(Chi_initial), np.imag(Chi_initial),
             np.real(theta_NS_initial), np.imag(theta_NS_initial), 
             np.real(theta_NF_initial), np.imag(theta_NF_initial)]
          
    gamma_NF_Steps = np.linspace(0,gamma_NF,StepNumber)
    EtaSteps = np.linspace(0,eta,StepNumber)
    
    for gammaIntermediate in gamma_NF_Steps:
        #Relax the gamma_NF = 0 condition
        Solution, Info, ErrorCheck, Message = fsolve(All_Equations,
            Guess, args=(Omega, 0, gamma_BNF, gammaIntermediate, gamma_BSN,
                  d_N1, xi_N, theta_S),
            full_output=True)
        
        if ErrorCheck == 0:
            raise RuntimeError(
                f"fsolve failed: {Message}\n"
                f"Omega={Omega}, gamma={gammaIntermediate}, "
                f"eta={0}\n"
                f"Guess={Guess}\n"
                f"Solution={Solution}")
        
        Guess = [Solution[0], Solution[1], 
                 Solution[2], Solution[3],
                 Solution[4], Solution[5]]
    
    for EtaIntermediate in EtaSteps:
        #Relax eta=0 condition
        Solution, Info, ErrorCheck, Message = fsolve(All_Equations,
            Guess,
            args=(Omega, EtaIntermediate, gamma_BNF, gamma_NF, gamma_BSN,
                  d_N1, xi_N, theta_S), 
            full_output=True)
        
        if ErrorCheck == 0:
            raise RuntimeError(
                f"fsolve failed: {Message}\n"
                f"Omega={Omega}, eta={EtaIntermediate}, "
                f"gamma_NF={gamma_NF}\n"
                f"Guess={Guess}\n"
                f"Solution={Solution}")
        
        Guess = [Solution[0], Solution[1], 
                 Solution[2], Solution[3],
                 Solution[4], Solution[5]]
    
    Chi_SNF = Solution[0] + 1j*Solution[1]
    
    return Chi_SNF

def JC_DiffuseExchange(d_F, Temperature, Resistivity_N, Resistivity_F, 
                       eta, CoherenceLength, H, gamma_NF, gamma_BSN, 
                       d_N1, d_N2, xi_N, SC_gap, Area, Amplitude=None, 
                       gamma_BNF=None, DeadLayer=0.0):
    """Calculate the critical voltage across the Josephson junction according
    to the Heim model.

    This function calculates the critical voltage, IcRn, as a function of 
    ferromagnetic thickness of the weak link as presented in the Eq 18 of 
    the paper by Heim et al: https://doi.org/10.1088/1367-2630/17/11/113022.
    The main complexity is the calculation of the two boundary constants, Chi, 
    which can represent either an SF junction, or an SNF junction, and two such
    Chi's for the left and right boundary will always need to be calculated.

    Args:
        d_F (numpy.ndarray): List of (float) thicknesses of the ferromagnetic 
            junction (nm).
        Temperature (float): Temperature of the system (K).
        Resistivity_N (float): Resistivity of the normal metal (Ohm*nm).
        Resistivity_F (float): Resistivity of the ferromagnet (Ohm*nm).
        eta (float): Spin-flip scattering parameter, defined as: eta = hbar/(pi*k_B*T_c*tau_m) 
            where tau_m is the spin-flip scattering time (unitless).
        CoherenceLength (float): Coherence length in the ferromagnet (nm).
        H (float): Exchange energy in the ferromagnet (eV).
        gamma_NF (float): Suppresion parameter between the normal metal
            and the superconductor (unitless).
        gamma_BSN (float): Boundary suppresion parameter between superconductor
            and normal metal (unitless).   
        d_N1 (float): Thickness of the normal metal at the left interface (nm).
        d_N2 (float): Thickness of the normal metal at the right interface (nm).
        xi_N (float): Coherence length in the normal metal (nm).    
        SC_gap (float): Superconducting gap (eV).
        Area (float): Area of the Josephson junction (nm^2).
        Amplitude (float): If provided, can be used to set an arbitrary scaled
            amplitude for the output.
        gamma_BNF (float): If provided, sets the boundary suppresion parameter 
            between the normal metal and the ferromagnet (unitless).
        DeadLayer (float): Thickness of dead (non-magnetic) material in the 
            ferromagnet. Negative values indicate increased effective 
            ferromagnetic thickness due to proximity magnetisation in the 
            normal metal.

    Returns:
        IcRn (float): Voltage across the Josephson junction (uV).

    Notes:

    """
    #Resistivity_F = (Resistivity_N*xi_N)/(gamma_NF*CoherenceLength)
    
    if Amplitude is None:
        Amplitude = Area*(16*np.pi*k_B*Temperature)/Resistivity_F #Area in nm^2
    
    d_F = d_F - DeadLayer
    
    J_c = np.zeros_like(d_F, dtype=np.complex128)
    
    N_list = np.arange(FreqCutoff)
    #"Omega" in this work will refer to Omega-tilda in the original paper
    Omega_list = (Temperature/T_c)*(2*N_list+1)+(H/(np.pi*k_B*T_c))*1j

    #eta = hbar/(np.pi*SpinScatterTime*k_B*T_c)
    if gamma_BNF is None:
        gamma_BNF = InterfaceResistance/(CoherenceLength*Resistivity_F)
    
    gamma_list = np.sqrt(Omega_list+eta)/CoherenceLength
    
    for gamma, w in zip(gamma_list, Omega_list):
        #Define theta_S from equation 5
        theta_S = np.arctan(SC_gap/(np.pi*k_B*T_c*np.real(w)))
        #Find the intial angles taking gamma_NF and eta = 0
        theta_NS_initial = Find_Theta_NS_Initial(d_N1, w, xi_N, gamma_BSN, theta_S)
        theta_NF_initial = Find_Theta_NF(d_N1, w, xi_N, theta_NS_initial, gamma_BSN, theta_S)
        
        theta_NS_initial2 = Find_Theta_NS_Initial(d_N2, w, xi_N, gamma_BSN, theta_S)
        theta_NF_initial2 = Find_Theta_NF(d_N2, w, xi_N, theta_NS_initial2, gamma_BSN, theta_S)
        
        #Exact solution of the quartic equation 20/22 and then selecting the real root
            
        Chi1 = Find_SNF_Boundary_Chi(gamma_BNF, w, theta_NF_initial, 
                                     theta_NS_initial, eta, theta_S, 
                                     gamma_NF, StepNumber)
        
        Chi2 = Find_SNF_Boundary_Chi(gamma_BNF, w, theta_NF_initial2, 
                                     theta_NS_initial2, eta, theta_S,
                                     gamma_NF, StepNumber)
               
        #Chi2 = Find_SF_Boundary_Chi(gamma_BSF, w, theta_S, eta, StepNumber)
        
        Term = np.real(gamma*np.exp(-gamma*d_F)*Chi1*Chi2)
      
        J_c += Term
        
    IcRn = JunctionResistance*Amplitude*np.abs(J_c) #V
        
    return IcRn*1E6 #Return the voltage in uV

#Load the data from the file Data.txt
#d,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T #units of nm, mA, mA

'''
d = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
              1.3, 2.45, 2.9, 3.05, 0.25, 0.3001, 0.35, 0.65, 0.75, 0.85, 3.35,
              3.5, 3.95, 4.1] )

y = np.array([65.340875, 24.878000000000004, 51.64000000000001, 61.19333333333333, 
              39.726, 25.309341886259293, 14.174717467474276, 18.957235235409815,
              13.972562686109798, 10.008982510530531, 6.939496641380081, 
              2.3033272230925714, 6.3283471086318865, 1.3283048839258265, 
              2.1937826740803543, 2.6960435520171293, 2.642885048481178, 
              26.75475, 21.053250000000006, 48.66025, 45.33733333333334, 
              11.033598167685819, 19.0885, 2.151310036922145, 5.327660826872811,
              2.3257632595999977, 2.2332393553653582])

dy = np.array([3.906590843129723, 0.7540000000000013, 2.8450014645573267, 
               1.514830390212418, 0.9546742245394503, 1.624217115530211,
               0.9081232687658485, 0.3261137420414092, 0.779596233884249,
               0.8897751989591207, 0.23563425164791063, 0.28325155792176143,
               0.4182558888900494, 0.26331238028429693, 0.21384513210250294,
               0.3840692332507965, 0.204401721667579, 0.7852499999999994,
               0.8842499999999979, 5.166759346614983, 5.5504418943199685,
               0.3490574302442736, 1.2972500000000018, 0.09114521005284514, 
               0.5093084657670612, 0.11055011898099443, 0.3681883271793756])
'''

d = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
              0.8, 0.85, 1.0,0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 0.15, 
              1.5, 1.65, 1.8])

y = np.array([25.485540677019983, 20.24388234661639, 9.229536446202395, 
              2.3917306615253358, 5.332381933490123, 6.262335355328344, 
              5.555634474421091, 7.14605149171436, 4.670040152958881, 
              4.466469736108299, 4.402734073344201, 2.8669437006283083, 
              2.198601946982659, 0.31666206152922616, 1.018208817810008, 
              6.094153299948496, 6.307979995277196, 5.485744923060322, 
              2.581500267199711, 0.3440655265061016, 1.3730251946938403,
              0.951189258094282, 0.08461134762633567, 38.13333333333333, 
              0.1905, 0.3, 0.12166666666666666])
dy = np.array([1.1541353059558421, 1.3282852847429805, 0.43555339620836153, 
               0.183136722622837, 0.32397334928648797, 0.16637832288545412, 
               0.4313857996461637, 0.2067104216127845, 0.26935663136737165, 
               0.11061774240230109, 0.5189157313556868, 0.13401844281653402,
               0.1469797151953465, 0.051007687876081016, 0.08432750457227471, 
               0.12104294939270999, 0.2824216946228165, 0.19224653742922326, 
               0.03978367734086671, 0.018540196895687144, 0.20333691412042817,
               0.0053777623606668535, 0.004048255991005446, 1.7975291683617007,
               0.0035000000000000027, 0.04999999999999999, 0.010137937550497038])

OrderingIndex = np.argsort(d)
d = d[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

#y = y/JunctionResistance
#dy = dy/JunctionResistance

Model = bmp.Curve(
    JC_DiffuseExchange,
    d, y, dy,
    Temperature=Temperature,
    #Resistivity_N = 87,#Ohm nm
    #Resistivity_F = Resistivity_F,
    gamma_NF=gamma_NF,
    gamma_BSN=gamma_BSN,
    d_N1=d_N1,
    d_N2=d_N2,
    xi_N=xi_N,
    SC_gap=SC_gap,
    CoherenceLength=CoherenceLength,
    #Amplitude = 100
    Area = Area,
    DeadLayer = DeadLayer
    )

### Limits of fitting values ###

#Model.CoherenceLength.range(0.2,3.5)
#Model.H.range(0.6,0.8)
#Model.Temperature.range(1,10)
#Model.eta.range(0,500)
Model.gamma_NF.range(1E-8,1E-2)
#Model.Resistivity_F.range(30,2000)
Model.gamma_BSN.range(0.01,3)
#Model.gamma_BNF.range(1.8, 2.5)
#Model.xi_N.range(5,60)
#Model.Resistivity_F.range(10000,100000)
#Model.Amplitude.range(100,3000)
#Model.DeadLayer.range(-0.5,-0.2)

#Model.CoherenceLength.dev(std=0.1, mean=0.3, limits=None)
#Model.SC_gap.dev(std=0.1, mean=0.3, limits=None)
#Model.Temperature.dev(std=0.1, mean=0.16, limits=None)
#Model.Resistance.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values

Model.CoherenceLength.value = CoherenceLength #nm
Model.H.value = H
Model.Temperature.value = Temperature
Model.eta.value = eta
Model.Resistivity_F.value =  Resistivity_F #Ohm nm
Model.Resistivity_N.value =  Resistivity_N #Ohm nm
Model.gamma_NF.value = gamma_NF
Model.SC_gap.value = 1.5E-3 #eV
Model.xi_N.value = xi_N #nm
Model.d_N1.value = 5 #nm
Model.d_N2.value = 10 #nm
Model.gamma_BSN.value = gamma_BSN
Model.Area.value = Area
Model.DeadLayer.value = DeadLayer
#Model.Amplitude.value = Amplitude

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot

plt.errorbar(
    d, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

#Resistivity_F = (Resistivity_N*xi_N)/(gamma_NF*CoherenceLength)
X_axis = np.linspace(0.1, 2, 100000)
J_0 = Area*np.pi*k_B*T_c/(Resistivity_F*CoherenceLength)

for test in [0.7]:
    ytest = JC_DiffuseExchange(
        X_axis,
        Temperature=4.2,
        Resistivity_N= Resistivity_N,#ohm nm,
        Resistivity_F=Resistivity_F, #ohm nm,
        CoherenceLength= CoherenceLength, #nm
        eta=eta,
        H=H, #0.6,#1.54468,#0.520934,
        gamma_NF= gamma_NF,
        gamma_BSN = gamma_BSN,#0.186,
        d_N1=5,
        d_N2=10,
        xi_N=xi_N,
        SC_gap = 1.5E-3, #eV
        Area = Area,
        #Amplitude = 0.001
        #gamma_BNF = 0.001,
        DeadLayer=DeadLayer
    )
    plt.plot(X_axis, ytest, label=f"gamma_NF {gamma_NF}", linewidth=3)
plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=34)
plt.legend(fontsize=34)
plt.xlabel("Thickness (nm)", fontsize=34)
plt.ylabel(r"$I_cR_N$ ($\mathrm{\mu V}$)", fontsize=34)
plt.show()
