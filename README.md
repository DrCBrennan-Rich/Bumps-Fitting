# Bumps-Fitting
This is a series of scripts for performing fitting with the Bumps package: https://bumps.readthedocs.io/en/stable/

Specifically implemetning the DREAM Markov Chain Monte Carlo (MCMC) methods: https://doi.org/10.1016/j.envsoft.2015.08.013

Currently the specific .py files exist for:

Fitting of a straight line - BumpsLinear.py

The enviroment file can be found in: requirements.txt

All fitting is ran through the commandline (the commandline input can be found commented at the top of each py file)

It will generally be of the form: "bumps -b --fit=dream --burn=200 --samples=1000 --init=random --export=BumpsOutput --session=Session.h5 BumpsScript.py"

bumps -b (run bumps in batch fitting mode) --fit=dream (use the DREAM MCMC method) --burn=200(disregard the first 200 results) --samples=1000(run 1000 trials after burn) --init=random (initialise variables randomly, other options are "best" and "eps") --export=BumpsOutput (Outputs files to a folder called "BumpsOutput" in directory) --session=Session.h5 (saves session file in director as "Session.h5) BumpsScript.py (runs the BumpsScript.py file)

Important! For proper graph output the MPLBACKEND must be set in the enviroment to prevent the matplotlib trying to produce GUI images:
set MPLBACKEND=SVG

#### Things to work on ####
Exporting both vector and raster images after fitting (the Bumps default are pngs) 
