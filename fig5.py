# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from numpy.fft import rfft, ifft, rfftfreq

clight = 2.99792458e10 # velocity of light / cm/s
Grav = 6.67259e-8      # grav const / cm^3/g/s^2 
Msun = 1.989e33        # solar mass / g
Mpc = 3.08568025e24    # 1Mpc / cm

def waveform(f, m1, m2, r=Mpc):
    """
    input:
      f = np.array of frequency / Hz
      m1,m2 = masses of coalescing binary stars / g
      r = distance from earth to binary stars / cm
    return:
      h = fourier transform of gravitational waves
    reference:
      B. Allen et al "FINDCHIRP" Physical Review D85(2012)122006
    """
    m = m1+m2
    eta = (m1 * m2)/m**2
    m_chirp = m * eta**0.6
    Mf = 8 * np.pi * Grav * m_chirp * f / clight**3
    beta = (np.pi * Grav * m * f)**(1/3) / clight
    alpha = 1 + beta**2 * (
        3715/756 + 55/9*eta
        + beta * (-16*np.pi
        + beta * (15293365/508032 + 27145/504*eta + 3085/72*eta**2)))
    f_low = 40
    f_isco = clight**3/(6**1.5 * np.pi * Grav * m)
    h = (5/3/np.pi)**0.5 * Grav * m_chirp / clight**2 / r
    h *= np.exp(-np.pi*0.25j + 0.75j / Mf**(5/3) * alpha)
    h /=  Mf**(1/6) * f
    h[f > f_isco] = 0
    h[f < f_low] = 0
    return h    

data = [np.loadtxt('H-H1_LOSC_4_V2-1126259446-32.txt'),
        np.loadtxt('L-L1_LOSC_4_V2-1126259446-32.txt') + 1.e-18]
label = ['Hanford', 'Livingston']
color = ['b', 'g']

(m1,m2) = (41.74, 29.24)
(t0,t1,t2) = (16.44, -16.44, 15.56)

fs = 4096
N = len(data[0])
freq = rfftfreq(N, 1/fs)

time = np.arange(N)/fs - t0
t = time[(time >= t1) & (time < t2)]
window = np.hanning(N)

plt.figure(figsize=(6.4, 5.4))

for (i,h) in enumerate(data):
    (S,f) = psd(h, Fs=fs, NFFT=N//8)
    S = np.interp(freq, f, S*fs)

    h = rfft(h * window)

    template = waveform(freq, m1*Msun, m2*Msun)*fs
    C = 2 * ifft(h * template / S, N)
    C = C[(time >= t1) & (time < t2)]
    sigma = np.sqrt(np.sum(np.abs(template)**2 / S) / N)
    rho = np.abs(C) / sigma
    rho_max = np.max(rho)

    print(label[i])
    print('SNR max =', rho_max)
    print('event time =', t[np.argmax(rho)] + t0)
    print('2*phi =', np.degrees(np.angle(C[np.argmax(rho)])))
    print('distance =', sigma/rho_max)

    plt.subplot(2, 1, i+1)
    plt.axis([t1, t2, 0, 11])
    plt.plot(t, rho, color[i], label=label[i], lw=1)
    plt.ylabel('signal to noise ratio', fontsize=14)
    plt.text(9, 8, '$m_1 = {:.2f}\,M_\odot$'.format(m1))
    plt.text(9, 7, '$m_2 = {:.2f}\,M_\odot$'.format(m2))
    plt.legend()

plt.xlabel('time since 2015/9/14/9:50:45.44  / sec', fontsize=14)
plt.tight_layout()
plt.savefig('fig5.eps')
plt.show()
