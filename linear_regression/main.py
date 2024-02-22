import functions
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2,2,101)

a0 = -2
a1 = -1

y = functions.reta(a0, a1, x)
yn = functions.ruido(y)

functions.plot_reta(x,y)
functions.plot_reta(x,yn)

mat = functions.solution_space(x,y)

functions.plot_solution_space(mat)