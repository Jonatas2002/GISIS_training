import functions
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2,2,101)
m_true = np.array([-2, -1])  
#a0 = -2
#a1 = -1

y = functions.build_polynomial_function(x, m_true)

noise_amp = 0.5  # AMPLITUDE DO RUIDO
yn = functions.add_noise(y, noise_amp)

functions.plot_reta(x,y)
functions.plot_reta(x,yn)
functions.plot_reta_ruido(x,y, yn)


a0 = np.linspace(-4,4,1001)
a1 = np.linspace(-5,5,1001)
mat = functions.solution_space(x, y, a0, a1, 1001)

functions.plot_solution_space(mat)