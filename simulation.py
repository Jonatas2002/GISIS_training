from modeling import scalar

def simulation():

    id = 0

    myWave = [scalar.Wavefield_1D(), 
              scalar.Wavefield_2D(),
              scalar.Wavefield_3D()] 

    # print(myWave[id]._type)
    myWave[id].get_type()

    myWave[id].set_wavelet()
    myWave[id].plot_wavelet()


if __name__ == "__main__":
    simulation()
    
if __name__ == "__main__":
    wave_1d = scalar.Wavefield_1D() 
    wave_1d.set_model()
    wave_1d.plot_model()




