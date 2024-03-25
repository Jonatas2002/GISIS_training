import numpy as np
import matplotlib.pyplot as plt


# Leitura do arquivo
nx = 282
nz = 1501
dt = 2e-3
dx = 25
t = np.arange(nz)*dt

cmp_gather = np.reshape(np.fromfile("signals_n_systems/arquivo binario/open_data_seg_poland_vibroseis_2ms_1501x282_shot_1.bin", dtype=np.float32), (nx, nz))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

perc = np.percentile(cmp_gather, 96)
# Vizualizando o CMP Gather
fig, ax = plt.subplots(num = "CMP Gather", figsize = (5,7))
ax.set_title('CMP Gather',  fontsize = 15)
ax.imshow(cmp_gather.T, aspect='auto', cmap='Grays', extent=[0, nx * dx, nz * dt, 0], vmin=-perc, vmax=perc)
ax.set_xlabel('x = offset[m]',  fontsize = 15)
ax.set_ylabel('t = TWT [s]',  fontsize = 15)
plt.show()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

trace1 = cmp_gather[0,:]
trace80 = cmp_gather[79,:]
trace161 = cmp_gather[160,:]

# Transformada de Fourier
freq = np.fft.fftfreq(nz, dt)
mascara = freq > 0
# ---------------------
Y1 = np.fft.fft(trace1)
Amp1 = np.abs(Y1 / nz)
# ---------------------
Y80 = np.fft.fft(trace80)
Amp80 = np.abs(Y80 / nz)
# ---------------------
Y161 = np.fft.fft(trace161)
Amp161 = np.abs(Y161 / nz)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(ncols = 1, nrows = 2, num = "TRAÇO 1", figsize = (15, 6))
ax[0].set_title('Time Domain', fontsize = 18)
ax[0].plot(t, trace1)
ax[0].set_xlabel("Time [s]", fontsize = 15)
ax[0].set_ylabel(r"$x(t)$", fontsize = 15)

ax[1].set_title("Fast Fourier Transform - FFT", fontsize = 18)
ax[1].plot(freq[mascara], Amp1[mascara])
ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)
ax[1].set_ylabel(r"$X(f)$", fontsize = 15)
ax[1].set_xlim(0,100)

fig.tight_layout()
plt.grid(axis = "y")
fig.savefig('signals_n_systems/imagens/fft_trace1.png')
plt.show()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(ncols = 1, nrows = 2, num = "TRAÇO 80", figsize = (15, 6))
ax[0].set_title('Time Domain', fontsize = 18)
ax[0].plot(t, trace80)
ax[0].set_xlabel("Time [s]", fontsize = 15)
ax[0].set_ylabel(r"$x(t)$", fontsize = 15)

ax[1].set_title("Fast Fourier Transform - FFT", fontsize = 18)
ax[1].plot(freq[mascara], Amp80[mascara])
ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)
ax[1].set_ylabel(r"$X(f)$", fontsize = 15)
ax[1].set_xlim(0,100)

fig.tight_layout()
plt.grid(axis = "y")
fig.savefig('signals_n_systems/imagens/fft_trace80.png')
plt.show()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(ncols = 1, nrows = 2, num = "TRAÇO 161", figsize = (15, 6))
ax[0].set_title('Time Domain', fontsize = 18)
ax[0].plot(t, trace161)
ax[0].set_xlabel("Time [s]", fontsize = 15)
ax[0].set_ylabel(r"$x(t)$", fontsize = 15)

ax[1].set_title("Fast Fourier Transform - FFT", fontsize = 18)
ax[1].plot(freq[mascara], Amp161[mascara])
ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)
ax[1].set_ylabel(r"$X(f)$", fontsize = 15)
ax[1].set_xlim(0,100)

fig.tight_layout()
plt.grid(axis = "y")
fig.savefig('signals_n_systems/imagens/fft_trace161.png')
plt.show()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

'''Trasnsformada Inversa de Fourier'''
# ---------------------
inverse_fourier = np.fft.ifft(Y1)

fig, ax = plt.subplots(ncols = 1, nrows = 2, num = "TRASNFORMADA INVERSA DE FOURIER - TRAÇO 1", figsize = (15, 6))
ax[0].set_title('Orignal Trace', fontsize = 18)
ax[0].plot(t, trace1)
ax[0].set_xlabel("Time [s]", fontsize = 15)
ax[0].set_ylabel(r"$x(t)$", fontsize = 15)

ax[1].set_title("Trace after inverse transformation", fontsize = 18)
ax[1].plot(t, inverse_fourier)
ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)
ax[1].set_ylabel(r"$X(f)$", fontsize = 15)

fig.tight_layout()
plt.grid(axis = "y")
fig.savefig('signals_n_systems/imagens/fft_trace1.png')
plt.show()

