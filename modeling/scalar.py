import numpy as np
import matplotlib.pyplot as plt

class Wavefield_1D():
    
    wave_type = "1D wave propagation in constant density acoustic isotropic media"
    
    def __init__(self):
        
        # TODO: read parameters from a file
        
        self.nt = 1001
        self.dt = 1e-3
        self.fmax = 30.0
        
        self.nz = 1000
        self.dz = 10
        self.model = np.zeros(self.nz)
        self.depth = np.arange(self.nz) * self.dz
        
        self.prof = np.array([2000, 4000, 6000, 8000, 10000])
        self.velocities = np.array((1500, 4000, 2000, 4500, 5000))
        
    def set_model(self):
        interfaces = []
        for i in range(len(self.prof)):
            start_depth = (self.prof[i] - self.prof[0]) / self.dz
            end_depth = self.prof[i] / self.dz
            vel = self.velocities[i]
            interfaces.append((start_depth, end_depth, vel))

        for j in range(len(interfaces)):
            start, end, vel = interfaces[j]
            start_depth = int(start)
            end_depth = int(end)
            self.model[start_depth:end_depth] = vel

    def plot_model(self):
        fig, ax = plt.subplots(figsize=(3, 6), clear=True)
        ax.plot(self.model, self.depth)
        ax.set_title("Velocity Model", fontsize=18)
        ax.set_xlabel("Velocity [m/s]", fontsize=15)
        ax.set_ylabel("Depth [m]", fontsize=15) 
        ax.set_ylim(max(self.depth), min(self.depth))

        fig.tight_layout()
        plt.show()   
               
    def get_type(self):
        print(self.wave_type)

    def set_wavelet(self):    
        
        t0 = 2.0*np.pi/self.fmax
        fc = self.fmax/(3.0*np.sqrt(np.pi))
        td = np.arange(self.nt)*self.dt - t0
        
        arg = np.pi*(np.pi*fc*td)**2.0
        
        self.wavelet = (1.0 - 2.0*arg)*np.exp(-arg)

    def plot_wavelet(self):      
         
        t = np.arange(self.nt)*self.dt
        
        fig, ax = plt.subplots(figsize = (10, 5), clear = True)
        
        ax.plot(t, self.wavelet)
        ax.set_title("Wavelet", fontsize = 18)
        ax.set_xlabel("Time [s]", fontsize = 15)
        ax.set_ylabel("Amplitude", fontsize = 15)    
             
        ax.set_xlim([0, np.max(t)])        
        
        fig.tight_layout()
        plt.show()
        

class Wavefield_2D(Wavefield_1D):
    
    def __init__(self):
        super().__init__()
        
        wave_type = "2D wave propagation in constant density acoustic isotropic media"    


class Wavefield_3D(Wavefield_2D):
    
    def __init__(self):
        super().__init__()
        
        wave_type = "3D wave propagation in constant density acoustic isotropic media"    
