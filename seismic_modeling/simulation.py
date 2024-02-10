from modeling import scalar
from time import time

def simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    # print(myWave[id]._type)
    myWave[id].get_type()

    myWave[id].set_wavelet()
    myWave[id].plot_wavelet()
    
    myWave[id].set_model()
    myWave[id].plot_model()
    
    start = time()
    myWave[id].wave_propagation()
    end = time()

    print(end - start)
    myWave[id].plot_wavefield()



if __name__ == "__main__":
    simulation()