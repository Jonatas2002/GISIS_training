import functions
import numpy as np
import matplotlib.pyplot as plt

depth = 500
velocity = 3000

t_0 = 2*depth/velocity

m_true = [(t_0)**2, (1/velocity)**2]   


N = 321  # NUMEROS DE PONTOS

xi = 50   # PONTO INICIAL
xf = 8050   # PONTO FINAL

x = np.linspace(xi, xf, N)  # VARIAVEL X

data_true = np.sqrt(functions.build_polynomial_function(x**2, m_true))    # calculo de m_true = Y = A + Bx + Cx² - Um array com mesma dimensão de x

#APLICANDO O RUIDO NO SINAL
noise_amp = 0.05  # AMPLITUDE DO RUIDO
data_noise = functions.add_noise(data_true, noise_amp)

m_calc = functions.least_squares_solver(x, data_noise, len(m_true))   # MMQ (X, data_noise)  Y = A + Bx + Cx²

data_calc = np.sqrt(functions.build_polynomial_function(x**2, m_true))    # m_true = Y = A + Bx + Cx²

# ---------------------------------------------------------
# ---------------------------------------------------------

vel = np.linspace(1500, 4500, N)
prof = np.linspace(250, 750, N)

mat = functions.solution_space2(vel, prof, x, data_noise)


# ---------------------------------------------------------
# ---------------------------------------------------------

fi, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (7,5))

ax.plot(x, data_noise, "o", markersize = 3)
ax.plot(x, data_calc)
ax.set_title('Polynomial Function')
ax.set_xlabel("Offset [m]")
ax.set_ylabel("TWT [s]")

ax.grid(True)

ax.invert_yaxis()
plt.tight_layout()
plt.savefig('linear_regression/polynomial_function.png')
plt.show()

# ---------------------------------------------------------
# ---------------------------------------------------------

min_ind = np.unravel_index(np.argmin(mat), mat.shape)
min_vel = np.linspace(1500,4500,N)[min_ind[1]]
min_prof = np.linspace(250,750,N)[min_ind[0]]
    

fi, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (7,7))

ax.imshow(mat, extent=[1500, 4500, 250, 750], origin='lower', aspect='auto')
ax.plot(min_vel, min_prof, color='red', marker='o')

ax.set_xlabel('Velocity space[m/s]')
ax.set_ylabel('Depth space [m]')
ax.set_title('Solution Space')

plt.tight_layout()
plt.savefig('linear_regression/Solution_Space.png')
plt.show()