import numpy as np
import matplotlib.pyplot as plt

domain = int(1e3)    # continuous function

t = np.linspace(-np.pi, np.pi, domain)

# Simple numerical series f(t) : R -> R

frequencies = np.array([2, 5, 8])  # Hz
dt = 1/32
signal = np.zeros(domain) 

for f in frequencies:
    signal += np.sin(2.0*np.pi*f*t)
    
    
# Transformada de Fourier do Sinal
freq = np.fft.fftfreq(domain, dt)
mascara = freq > 0
Y = np.fft.fft(signal)
Amp = np.abs(Y / domain)

# Plot 

xloc = np.linspace(-np.pi, np.pi, 9)

xlab = [r"$-\pi$", r"$\dfrac{-3\pi}{4}$", r"$\dfrac{-\pi}{2}$", r"$\dfrac{-\pi}{4}$", 
        r"$0$", r"$\dfrac{\pi}{4}$", r"$\dfrac{\pi}{2}$", r"$\dfrac{3\pi}{4}$", r"$-\pi$"]

fig, ax = plt.subplots(num = "Simple signal", figsize = (15,5))

ax.plot(t, signal)

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)
ax.set_xlim([-np.pi, np.pi])

ax.set_xlabel("t [s]", fontsize = 15)
ax.set_ylabel("Amplitude", fontsize = 15)

fig.tight_layout()
plt.show()


fig, ax = plt.subplots(num = "Simple signal - FFT", figsize = (15,5))

ax.plot(freq[mascara], Amp[mascara])

ax.set_xlabel("Frequencia", fontsize = 15)
ax.set_ylabel("Amplitude", fontsize = 15)

fig.tight_layout()
plt.show()

# --------------------------------------------------
# --------------------------------------------------

title = "First example"

a = 2.0 * np.sinh(np.pi)/np.pi

signal = a + np.zeros(domain)

for n in range(1,501):
    
    c = a*(-1)**n / (1 + n**2) 

    signal += c*np.cos(n*t) + n*c*np.sin(n*t)

fig, ax = plt.subplots(num = title, figsize = (15,5))

ax.plot(t, signal - np.pi, label = "Fourier series")
ax.plot(t, np.exp(-t), "--", label = "Real function")

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)
ax.set_xlim([-np.pi, np.pi])

ax.set_xlabel("t [s]", fontsize = 15)
ax.set_ylabel("Amplitude", fontsize = 15)

ax.legend(loc = "upper left", fontsize = 15)

fig.tight_layout()
plt.show()

# --------------------------------------------------
# --------------------------------------------------

title = "Second example"

signal2 = np.zeros(domain)

for n in range(1, 50):
    signal2 += (-2*np.sin(n*t)/n) * ((-1)**n + (np.sin(n*np.pi)/n*np.pi))

fig, ax = plt.subplots(num = title, figsize = (15,5))

ax.plot(t, signal2 - np.pi, label = "Fourier series")
ax.plot(t,t - np.pi, "--", label = "Real function")

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)
ax.set_xlim([-np.pi, np.pi])
ax.grid()
ax.set_xlabel("t [s]", fontsize = 15)
ax.set_ylabel("Amplitude", fontsize = 15)

ax.legend(loc = "upper left", fontsize = 15)

fig.tight_layout()
plt.show()

# --------------------------------------------------
# --------------------------------------------------

title = "Third example"

a = (2.0 * np.pi**2)/3

signal3 = a + np.zeros(domain)

for n in range(1,50):
    
    signal3 += ((4 * (-1)**(n + 1))/n**2) * np.cos(n*t)
    
fig, ax = plt.subplots(num = title, figsize = (15,5))

ax.plot(t, signal3 - np.pi, label = "Fourier series")
ax.plot(t, -t**2 + 2*np.pi, "--", label = "Real function")

ax.set_xticks(xloc)
ax.set_xticklabels(xlab)
ax.set_xlim([-np.pi, np.pi])
ax.grid()
ax.set_xlabel("t [s]", fontsize = 15)
ax.set_ylabel("Amplitude", fontsize = 15)

ax.legend(loc = "upper left", fontsize = 15)

fig.tight_layout()
plt.show()

