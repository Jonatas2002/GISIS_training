import numpy as np
import matplotlib.pyplot as plt

shots = np.loadtxt('geometria/geometria2/coord_src.txt', dtype=float, skiprows=1, delimiter=',')
station = np.loadtxt('geometria/geometria2/coord_rec.txt',  dtype=float,skiprows=1, delimiter=',')
relation = np.loadtxt('geometria/geometria2/relational.txt', dtype=int, skiprows=1, delimiter=',')

shots_x = np.sqrt(shots[:,0]**2 + shots[:,1]**2)
station_x = np.sqrt(station[:,0]**2 + station[:,1]**2)

"""spread = relation[0,1] - relation[0,0]

CMP = np.zeros((spread*len(shots)))

for i in range(len(shots)):
    CMP[i*spread:i*spread + spread] = shots[i] - 0.5*(shots[i] - station[relation[i,0]:relation[i,1]])

CMPx,  CMPt = np.unique(CMP, return_counts= True)"""


plt.figure(figsize=(10,2))
plt.plot(station[:,1], station[:,0], 'o')
plt.plot(shots[:,1], shots[:,0], 'o')
#plt.plot(relation, np.zeros(len(relation)), 'o', alpha = 0.1, markersize=5)
plt.show()

"""plt.figure()
plt.plot(CMPx, CMPt)
plt.show()
"""

