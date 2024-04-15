import numpy as np
import matplotlib.pyplot as plt

shots = np.loadtxt('geometria/shots.txt', dtype=float, skiprows=1)
station = np.loadtxt('geometria/stations.txt',  dtype=float,skiprows=1)
relation = np.loadtxt('geometria/relation.txt', dtype=int, skiprows=1, delimiter=',')

spread = relation[0,1] - relation[0,0]

CMP = np.zeros((spread*len(shots)))

for i in range(len(shots)):
    CMP[i*spread:i*spread + spread] = shots[i] - 0.5*(shots[i] - station[relation[i,0]:relation[i,1]])

CMPx,  CMPt = np.unique(CMP, return_counts= True)

plt.figure()
plt.plot(station, np.zeros(len(station)), 'o')
plt.plot(shots, np.zeros(len(shots)), 'o')
plt.plot(relation, np.zeros(len(relation)), 'o', alpha = 0.1, markersize=5)
plt.show()

plt.figure()
plt.plot(CMPx, CMPt)
plt.show()


