import numpy as np
import matplotlib.pyplot as plt

shots = np.loadtxt('geometria/shots.txt', skiprows=1)
station = np.loadtxt('geometria/stations.txt', skiprows=1)
relation = np.loadtxt('geometria/relation.txt', skiprows=1, delimiter=',')

receiver_spacing = 10        
receivers_per_shot = 96

total_shots = len(shots)       
#near_offset = 10    
shot_spacing = 10



