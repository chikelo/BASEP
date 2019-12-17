# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:56:16 2019

@author: Eric
"""


#Imports
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize as opt
import pandas as pd
from pandas import DataFrame, Series

import scipy.optimize as spo


def trackingLoading(filePath):
    t = pd.read_csv(filePath ,sep = "\t", index_col=0)
    t = t.filter(["x","y","frame", "particle"])
    return t

def tmstmpLoading(filePath):
    timestamp = pd.read_csv(filePath, sep="\r", header = None,names=["time"])

    #Calculation of timestamps in nanoseconds
    timestamp["datetime"] =  pd.to_datetime(timestamp.time+2*3600, unit="s", origin=pd.Timestamp('1904-01-01'))
    timestamp["ellapsed_time"] = (timestamp.datetime-timestamp.datetime[0])
    timestamp["time"] = timestamp.ellapsed_time.dt.total_seconds()
    return timestamp

def get_center(trj):
    
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((trj.x.values-xc)**2 + (trj.y.values-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()    

    center_estimate = 0, 0
    center, ir = spo.leastsq(f_2, center_estimate)
    return center

def getPolarCoordinates(t):
    center = get_center(t)
    t["x0"] = t.x-center[0]
    t["y0"] = t.y-center[1]
    t["r"] = np.sqrt(t.x0**2 + t.y0**2)
    t["theta"] = np.arctan2(t.y0,t.x0)
    return t

def changeReferenceSystem(t, timestamp, omega):
    
    t["theta_prime"] = t.theta-omega*timestamp
    t["theta_prime_unwrap"] = np.unwrap(t.theta_prime)
    t["d_theta_prime"] = t.theta_prime_unwrap.diff()
    t["theta_dot_prime"] = t.d_theta_prime/timestamp.diff()
    return t

def currentDensityObtainment(t, timestamp, particles, part_density, omega):

    theta_prime= pd.DataFrame()
    omega_particle = pd.DataFrame()
    omega_particle_mean = pd.DataFrame()
    
    for p in particles:
        theta_prime[p] =abs(np.unwrap(t.loc[t.particle==p].theta)-timestamp.time*omega)


    for p in particles:
        omega_particle[p] = theta_prime[p].diff()/timestamp.time.diff()
    
    omega_particle_mean = omega_particle.mean(axis=1)
    
    vel_mean = omega_particle_mean.mean()
    current_density = vel_mean*part_density
    return current_density

"""
#Loading the tracking matrices from csv file 
newFilepath = []

#Defining first file name to load
filePath = list("Tracking_Test1_20181219.dat")

#Loading each file name in a single array of strings (one position for each complete name)
for tracking in np.linspace(1,7,7):
    filePath[13] = str(int(tracking))
    newFilepath.append("".join(filePath))
"""

newFilepath = ["Tracking_Test2_20181219.dat",
               "Tracking_Test3_20181219.dat",
               "Tracking_Test1_20181219.dat",
               "Tracking_Test4_20181219.dat",
               "Tracking_Test5_20181219.dat",
               "Tracking_Test6_20181219.dat",
               "Tracking_Test7_20181219.dat"]

timestampsFilepath = ["C:/Users/Eric/Desktop/BASEP_tests/20181219/Test2_2018_12_19_13_19_55.dat", 
                      "C:/Users/Eric/Desktop/BASEP_tests/20181219/Test3_2018_12_19_13_06_08.dat", 
                      "C:/Users/Eric/Desktop/BASEP_tests/20181219/Test1_2018_12_19_12_34_22.dat",
                      "C:/Users/Eric/Desktop/BASEP_tests/20181219/Test4_2018_12_19_13_36_33.dat",
                      "C:/Users/Eric/Desktop/BASEP_tests/20181219/Test5_2018_12_19_13_49_05.dat",
                      "C:/Users/Eric/Desktop/BASEP_tests/20181219/Test6_2018_12_19_15_26_32.dat",
                      "C:/Users/Eric/Desktop/BASEP_tests/20181219/Test7_2018_12_19_15_43_12.dat"]

Nparticles = np.array([27, 26, 24, 15, 9, 6, 3])
particleDens = Nparticles/(2*np.pi*4.5)
currentDensity = []

for i in np.linspace(0,6,7, dtype=int):
    
    t = trackingLoading(newFilepath[i])
    time = tmstmpLoading(timestampsFilepath[i])
    
    particles = t['particle'].unique().tolist()

    
    t = getPolarCoordinates(t)
    currentDensity.append(currentDensityObtainment(t, time, particles, particleDens[i], 0.4))
    
print(currentDensity)

plt.plot(particleDens, currentDensity)



