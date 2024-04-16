import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Leitura do arquivo
nx = 282
nz = 1501
dt = 2e-3
dx = 25
t = np.arange(nz)*dt

cmp_gather = np.reshape(np.fromfile("signals_n_systems/arquivo binario/open_data_seg_poland_vibroseis_2ms_1501x282_shot_1.bin", dtype=np.float32), (nx, nz))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

trace1 = cmp_gather[0,:]
trace80 = cmp_gather[79,:]
trace100 = cmp_gather[99,:]
trace161 = cmp_gather[160,:]

# Transformada de Fourier
freq = np.fft.fftfreq(nz, dt)
mascara = freq > 0

# ---------------------
Amp1 = np.abs(np.fft.fft(trace1) / nz)
# ---------------------
Amp80 = np.abs(np.fft.fft(trace80) / nz)
# ---------------------
Amp100 = np.abs(np.fft.fft(trace100) / nz)
# ---------------------
Amp161 = np.abs(np.fft.fft(trace161) / nz)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#perc = np.percentile(cmp_gather, 95)
perc = 95
perc_min = np.percentile(cmp_gather, 100-perc)
perc_max = np.percentile(cmp_gather, perc)

# Vizualizando o CMP Gather


# Crie os subplots com tamanhos diferentes
# Defina os tamanhos dos subplots
widths = [5, 2]  # largura do primeiro e segundo subplot, respectivamente
heights = [7]  # altura do subplot único (1 linha)

# Crie uma grade de subplots com tamanhos diferentes
fig = plt.figure(figsize=(sum(widths), max(heights)))
gs = gridspec.GridSpec(len(heights), len(widths), width_ratios=widths, height_ratios=heights)

# Adicione os subplots
ax0 = plt.subplot(gs[0, 0])
ax1 = plt.subplot(gs[0, 1])

#fig, ax = plt.subplots(ncols=2, nrows=1, num = "CMP Gather", figsize = (5,7))
ax0.set_title('CMP Gather',  fontsize = 15)
ax0.imshow(cmp_gather.T, aspect='auto', cmap='Grays', extent=[-(nx * dx)/2, (nx * dx)/2, nz * dt, 0], vmin=perc_min, vmax=perc_max)
ax0.plot(trace100 - 1025, t, 'red', '---')

ax0.set_xlabel('x = offset[m]',  fontsize = 15)
ax0.set_ylabel('t = TWT [s]',  fontsize = 15)
ax0.set_xticks(np.arange(-3525,3526,1175))

ax1.set_title("Trace 100 - FFT", fontsize = 15)
ax1.plot(Amp100, freq)
ax1.set_xlabel("Frequency [Hz]", fontsize = 11)
ax1.set_ylabel(r"$X(f)$", fontsize = 11)
ax1.set_ylim(100,-100)

fig.tight_layout()
fig.savefig('signals_n_systems/imagens/fft_trace100.png')
plt.show()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def calc_freq_med(DATA, dt): 

  DADO_f = 0
  for i in range(DATA.shape[1]):
    temp = np.fft.fftshift(np.fft.fft(DATA[:,i]))
    DADO_f += np.abs(temp)/np.max(np.abs(temp))

  DADO_f = DADO_f/DATA.shape[1]
  DADO_f = DADO_f/np.max(np.abs(DADO_f))
  FREQSf = np.fft.fftshift(np.fft.fftfreq(DATA.shape[0], d = dt))

  return DADO_f, FREQSf

dado_f, freqsf = calc_freq_med(cmp_gather, dt)

plt.figure(dpi=100)
plt.title('Espectro de Frequências', weight='bold', fontsize=16)
plt.plot(freqsf[cmp_gather.shape[0]//2:], np.abs(dado_f[cmp_gather.shape[0]//2:]))
plt.xlabel('Frequências (Hz)')
plt.ylabel('Amplitude Normalizada')
plt.grid(linestyle='--')
plt.show()

dado_FK = np.fft.fftshift(np.fft.fft2(cmp_gather)) # Calculando a transformada FK e reorganizando o array
freqsk = np.fft.fftshift(np.fft.fftfreq(dado_FK.shape[1], d = dx)) # Calculando os números de onda que representam o dado e reorganizando o array

perc = np.percentile(np.abs(dado_FK), 95)
plt.figure(dpi=100)
plt.title('Espectro FK', weight='bold', fontsize=16)
plt.imshow(np.abs(dado_FK), vmin=0, vmax=perc, aspect='auto', cmap='jet', extent = [freqsk[0], freqsk[-1], freqsf[0], freqsf[-1]])
plt.xlabel('Número de onda k')
plt.ylabel('Frequência (Hz)')
plt.colorbar()
plt.show()


print(np.shape(dado_FK))