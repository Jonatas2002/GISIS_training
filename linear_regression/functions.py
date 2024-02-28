import numpy as np
import matplotlib.pyplot as plt

# função que descreve a reta
def build_polynomial_function(x, parameters):

    function = 0.0
    for n, p in enumerate(parameters):  # A função enumerate gera um loop de elementos definidos pelo usuario
        function += p*x**n    # nesse caso, um loop de p*x^0 até p*x^n  (# a0 + a1.x + a2.x² + a3.x³ + ...)

    return function

# aplicar o ruido no eixo y
def add_noise(data, noise_amp):
    return data + noise_amp*(np.random.rand(len(data)))

# visualização da reta
def plot_reta(x,y):
	fig, ax = plt.subplots()
	ax.plot(x,y)
	
	fig.tight_layout()	
	plt.show()
 
# visualização da reta
def plot_reta_ruido(x,y, yn):
	fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (7,5))
	ax.plot (x,y,label='y limpo')
	ax.plot (x,yn, '-', label='y corrompido')
 
	fig.tight_layout()	        
	fig.savefig('linear_regression/RetaRuido.png')
	plt.show()

# criar espaço solução com varios coeficientes a0 e a1
# correlacionar atraves da norma L2 a diferença
def solution_space(x,y, a0, a1, n):
	
	#n = 1001
	
	#a0 = np.linspace(-4,4,n)
	#a1 = np.linspace(-5,5,n)
	
	a0, a1 = np.meshgrid(a0,a1)
	
	mat = np.zeros((n,n))
	
	for i in range(n):
		for j in range(n):	
			y_p = a0[i,j] + a1[i,j]*x	
	
			mat[i,j] = np.sqrt(np.sum((y - y_p)**2))  
		
	return mat

def solution_space2(velocity, depth, x, y):
    mat2 = np.zeros((len(depth),len(velocity)))

    for i in range(len(depth)):
        for j in range(len(velocity)):
            y_p = np.sqrt((x**2 + 4*depth[i]**2) / velocity[j]**2)
            
            mat2[i, j] = np.sqrt(np.sum((y - y_p)**2))

    return mat2

# plotar o espaço solução
def plot_solution_space(mat):
    min_ind = np.unravel_index(np.argmin(mat), mat.shape)
    min_a0 = np.linspace(-5,5,1001)[min_ind[1]]
    min_a1 = np.linspace(-5,5,1001)[min_ind[0]]
    
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (7,7))
    ax.imshow(mat, extent = [-5,5,-5,5])
    ax.plot(min_a0, -min_a1, color='red', marker='o')
    
    fig.tight_layout()	
    fig.savefig('linear_regression/SolutionSpace.png')

    plt.show()
        

def least_squares_solver(x, d, M):

    G = np.zeros((len(d), M))   # Criação de uma matriz G de len(d) linhas e M coluna com

    for n in range(M):
        G[:,n] = x**n          # Adicionando valores na matriz

    GTG = np.dot(G.T, G)       # G.T matriz transposta de G multiplicada pela matriz G
    GTd = np.dot(G.T, d)       # G.T matriz transposta de G multiplicada pela matriz d

    return np.linalg.solve(GTG, GTd)    # solution of a linear system

