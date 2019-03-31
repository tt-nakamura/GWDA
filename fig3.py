# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from numpy.fft import rfft, irfft, rfftfreq

data = [np.loadtxt('H-H1_LOSC_4_V2-1126259446-32.txt'),
        np.loadtxt('L-L1_LOSC_4_V2-1126259446-32.txt') + 1.e-18]
label = ['Hanford', 'Livingston']

fs = 4096
(t0,t1,t2) = (16.44, -0.5, 0.5)

N = len(data[0])
time = np.linspace(0, 32, N, endpoint=False) - t0
freq = rfftfreq(N, 1/fs)

NFFT = fs // 16
noverlap = NFFT // 16 * 15

plt.figure(figsize=(6.4, 7.4))

for (i,h) in enumerate(data):
    (S,f) = psd(h, Fs=fs, NFFT=N//8)
    S = np.interp(freq, f, S)
    h = irfft(rfft(h)/np.sqrt(S/2*fs))
    h = h[(time >= t1) & (time < t2)]

    plt.subplot(2, 1, i+1)
    plt.specgram(h, NFFT=NFFT, Fs=fs, noverlap=noverlap,
                 cmap='nipy_spectral', xextent=[t1,t2], vmin=-50, vmax=-10)
    plt.axis([t1, t2, 0, 500])
    plt.colorbar()
    plt.title(label[i], fontsize=14)
    plt.ylabel('frequency  / Hz', fontsize=14)

plt.xlabel('time since 2015/9/14/9:50:45.44  / sec', fontsize=14)
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
