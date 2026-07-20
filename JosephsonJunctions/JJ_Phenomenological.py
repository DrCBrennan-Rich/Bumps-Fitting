# -*- coding: utf-8 -*-
"""
@author: pycbr
"""
#Equation (6) from https://doi.org/10.1063/5.0195229
#### Run with: bumps -b --fit=dream --burn=1000 --samples=10000 --init=random --export=ExportFolder --session=JJSession.h5 JJ_Phenomenological.py

import bumps.names as bmp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})

#Coherence lengths
CoherenceLength_F1=0.3 #nm
CoherenceLength_F2=0.16 #nm

def JC_model(d_F, Amplitude, CoherenceLength_F1, CoherenceLength_F2, d_0pi):
    
    SinTerm = np.sin((d_F-d_0pi)/CoherenceLength_F2)
    
    return Amplitude*(np.exp(-d_F/CoherenceLength_F1)*np.abs(SinTerm))

#Load the data from the file Data.txt
#d_F,y,dy = np.loadtxt('PtCoPt data 4.2K.txt').T

d_F = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0, 0.30000000000000004, 0.44999999999999996, 0.6000000000000001, 0.75, 0.8999999999999999, 1.0499999999999998, 1.2000000000000002, 1.35, 0.15000000000000002, 1.5, 1.6500000000000001, 1.7999999999999998])
y = np.array([np.float64(25.485540677019983), np.float64(20.24388234661639), np.float64(9.229536446202395), np.float64(2.3917306615253358), np.float64(5.332381933490123), np.float64(6.262335355328344), np.float64(5.555634474421091), np.float64(7.14605149171436), np.float64(4.670040152958881), np.float64(4.466469736108299), np.float64(4.402734073344201), np.float64(2.8669437006283083), np.float64(2.198601946982659), np.float64(0.31666206152922616), np.float64(1.018208817810008), np.float64(6.094153299948496), np.float64(6.307979995277196), np.float64(5.485744923060322), np.float64(2.581500267199711), np.float64(0.3440655265061016), np.float64(1.3730251946938403), np.float64(0.951189258094282), np.float64(0.08461134762633567), np.float64(38.13333333333333), np.float64(0.1905), np.float64(0.3), np.float64(0.12166666666666666)])
dy = np.array([np.float64(1.1541353059558421), np.float64(1.3282852847429805), np.float64(0.43555339620836153), np.float64(0.183136722622837), np.float64(0.32397334928648797), np.float64(0.16637832288545412), np.float64(0.4313857996461637), np.float64(0.2067104216127845), np.float64(0.26935663136737165), np.float64(0.11061774240230109), np.float64(0.5189157313556868), np.float64(0.13401844281653402), np.float64(0.1469797151953465), np.float64(0.051007687876081016), np.float64(0.08432750457227471), np.float64(0.12104294939270999), np.float64(0.2824216946228165), np.float64(0.19224653742922326), np.float64(0.03978367734086671), np.float64(0.018540196895687144), np.float64(0.20333691412042817), 0.0053777623606668535, 0.004048255991005446, np.float64(1.7975291683617007), np.float64(0.0035000000000000027), np.float64(0.04999999999999999), np.float64(0.010137937550497038)])

OrderingIndex = np.argsort(d_F)
d_F = d_F[OrderingIndex]
y = y[OrderingIndex]
dy = dy[OrderingIndex]

y = y/1.9E-3
dy = dy/1.9E-3

Model = bmp.Curve(
    JC_model,
    d_F, y, dy,
    Amplitude=10000, 
    CoherenceLength_F1=0.3, 
    CoherenceLength_F2=0.16, 
    d_0pi=1.0)

### Limits of fitting values ###

Model.Amplitude.range(20/1.9E-3,120/1.9E-3)
Model.d_0pi.range(0.0,0.9*np.pi*CoherenceLength_F2) #Due to the periodicity of d_0pi, this will be the paramter range
Model.CoherenceLength_F1.range(0.1,0.6)
Model.CoherenceLength_F2.range(0.1,0.2)

#Model.CoherenceLength_F1.dev(std=0.1, mean=0.3, limits=None)
#Model.CoherenceLength_F2.dev(std=0.1, mean=0.16, limits=None)

#######
#Initial values

Model.Amplitude.value = 120000
Model.d_0pi.value = 0.35
Model.CoherenceLength_F1.value = 0.4
Model.CoherenceLength_F2.value = 0.17

problem = bmp.FitProblem(Model)

#This line is not strictly required, but allows you to run this py file check the initial parameters.
problem.show()

#Run some test values to see how they affect the final plot

plt.errorbar(
    d_F, y, yerr=dy,
    fmt='H',
    capsize=3,
    label='Experimental data')

X_axis = np.linspace(0.1, 2, 1000)

for d_0pi_test in [0.361262]:
    ytest = JC_model(
        X_axis,
        Amplitude= 23979.9, 
        CoherenceLength_F1= 0.311356,
        CoherenceLength_F2=0.163457, 
        d_0pi=d_0pi_test, 
    )
    plt.plot(X_axis, ytest, label=f"Fitted Curve", linewidth=3)
    
plt.yscale("log")
plt.xlabel("Thickness (nm)")
plt.ylabel("Current (mA)")
plt.legend()
plt.show()
