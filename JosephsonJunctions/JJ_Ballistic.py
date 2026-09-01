# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (1) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=Export --session=JJSession.h5 JJ_Ballistic.py
'''
#### WARNING ####

The function JC_model has been written to be fast and easily callable thousands
of times. This is at the cost of stability if the ferromagnetic coherence
length is smaller than 0.2 nm (see associated Figure on GitHub):
https://github.com/DrCBrennan-Rich/Bumps-Fitting/blob/main/JosephsonJunctions/Ballistic_CoherenceLengthComparison.pdf
And so this should be set as the lower fitting bound.
In situations where the coherence length is approaching this limit, it would be advisable to use:
https://github.com/DrCBrennan-Rich/Bumps-Fitting/blob/main/JosephsonJunctions/JJ_Ballistic_simplified.py
'''

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy import constants

k_B = constants.physical_constants['Boltzmann constant in eV/K'][0] #eV/K

T = 5 #Temperature in K
alpha = 1
y_max = 10*alpha

JunctionResistance = 1.4E-3 #ohms
T = 5 #K
SC_gap = 1.5E-3 #eV

Beta = 1/(k_B*T)

def JC_model(Thickness, CoherenceLength, SC_gap):
    
    #Produce a list of phase differences across the junction.
    PhiList = np.linspace(0,2*np.pi,60,endpoint=False)[:,None]
    
    #Produces a list of alpha values with minumum value 1E-6
    AlphaList = np.maximum(Thickness/CoherenceLength, 1E-6)
    
    IcRn = np.zeros_like(AlphaList)
    
    for i,Alpha in enumerate (AlphaList):
        
        yMax = max(10*Alpha, 5)
        Step = max(200, int(100*yMax))
        yList = np.linspace(Alpha, yMax, Step)[None,:]
        
        SinMinus = np.sin(0.5*(PhiList-yList))
        SinPlus = np.sin(0.5*(PhiList+yList))
        
        TanhMinus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList - yList)))
        TanhPlus = np.tanh(0.5*Beta*SC_gap*np.cos(0.5*(PhiList + yList)))
        
        Integrand = (1/(yList*yList*yList))*(SinMinus*TanhMinus+SinPlus*TanhPlus)
        
        Iphi = np.pi*SC_gap*Alpha*Alpha*trapezoid(Integrand, x=yList, axis=1)/2
        
        IcRn[i] = np.max(np.abs(Iphi))
        
    return 1E-6*IcRn #Voltage in uV
        

#Load the data from the file Data.txt
#d,y,dy = np.loadtxt('L11 data 4.2K.txt').T

d = np.array([2.418, 2.359, 2.339, 2.29, 2.251, 2.123, 2.103, 2.083, 1.955, 1.936, 
1.916, 1.827, 1.788, 1.749, 1.64, 1.621, 1.552, 1.532, 1.345, 1.404, 
1.384, 1.158, 1.069, 1.03, 1.01, 0.961, 0.922, 0.863, 0.774, 0.735, 
0.715, 0.627, 0.607, 0.627, 0.568, 0.459, 0.44, 0.42, 0.479, 0.459, 
0.371, 0.351, 0.331, 0.371, 0.331, 0.272])

y = np.array([0.014280000000000003, 0.020475000000000004, 0.054180000000000006, 
0.036375000000000005, 0.032625, 0.113685, 0.149295, 0.16805249999999997,
 0.19130888902289456, 0.24452465349230523, 0.3981007515000644, 
0.27330691468911794, 0.30633750000000004, 0.2771325, 0.1353852872781118,
 0.14751657583320804, 0.02249298790820722, 0.04697036769544414, 
0.3021027076445733, 0.43569413478486935, 0.5929766953508618, 
1.9408926118160599, 4.864723927574347, 6.2805, 5.816249999999999, 
9.014867286021701, 9.877036997792182, 11.293528114853833, 21.13536, 
22.281666666666666, 23.435604444444444, 19.214924999999997, 20.5390625,
 19.214924999999997, 11.239675337812473, 2.0448195943046024, 6.773110338724988,
 8.23716754905705, 2.873020383233309, 2.0448195943046024, 19.314, 
9.027999999999999, 26.774500000000003, 19.314, 26.774500000000003, 42.28124999999999])

dy = np.array([0.018272242610035586, 0.009652590325917705, 0.008517370486247504, 0.01028992225432243,
 0.005367727638395974, 0.0037760097987161027, 0.016000049218674296, 0.005118781593309099, 
0.006688453618545094, 0.011916942566567142, 0.03675846921195906, 0.005348239951696597,
 0.0036130769435482584, 0.0016652702483380903, 0.004372225594834334, 0.003198218846597834, 
0.002908114429761657, 0.0009786784239315836, 0.009785703920974102, 0.0014727845029029463,
 0.017693955583427543, 0.013422598382096257, 0.09176762360748456, 0.3372573201577695,
 0.09015680784056161, 0.12343506655023133, 0.17938964210720848, 0.23754582433868957, 
1.0612181378020253, 0.5142812676174606, 0.37071555555555763, 3.7148250000000003,
 0.4703125000000021, 3.7148250000000003, 0.5076082891649284, 0.6476795050413368, 
0.048506008277787734, 0.051299679274587784, 0.11229927064527502, 0.6476795050413368, 
6.067999999999999, 2.3280549821685907, 11.160499999999999, 6.067999999999999, 
11.160499999999999, 2.3353312131258814])


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

#Plot data points if desired

plt.errorbar(
    d, y, yerr=dy,
    fmt='o',
    capsize=3,
    label='Experimental data'
)

x_axis = np.linspace(0.01,10,100)

for CoherenceLength_test in [5E-1]:
    ytest = JC_model(
        x_axis,
        CoherenceLength=CoherenceLength_test,
        SC_gap = SC_gap)
    
    plt.plot(x_axis, ytest, label=f"Coherence Length={CoherenceLength_test}")
    plt.yscale('log')

plt.xlabel("Ferromagnet thickness (nm)")
plt.ylabel(r"Critical current density $J_c$ (mA/m$^2$)")

plt.legend()
plt.savefig("CoherenceLengthComparison.svg", format="svg")
plt.show()
