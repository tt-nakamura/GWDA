# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import butter, filtfilt

data = [np.loadtxt('H-H1_LOSC_4_V2-1126259446-32.txt'),
        np.loadtxt('L-L1_LOSC_4_V2-1126259446-32.txt') + 1.e-18]
label = ['Hanford', 'Livingston']
color = ['b', 'g']

fs = 4096
(f_min, f_max) = (43, 300)
(t0,t1,t2) = (16.44, -0.15, 0.05)

N = len(data[0])
time = np.linspace(0, 32, N, endpoint=False) - t0
freq = rfftfreq(N, 1/fs)
t = time[(time >= t1) & (time < t2)]

(b,a) = butter(4, [f_min*2/fs, f_max*2/fs], btype='band')

plt.figure(figsize=(6.4, 5))

for (i,h) in enumerate(data):
    (S,f) = psd(h, Fs=fs, NFFT=N//8)
    S = np.interp(freq, f, S)
    h = irfft(rfft(h)/np.sqrt(S/2*fs))
    h = filtfilt(b,a,h)
    h = h[(time >= t1) & (time < t2)]

    plt.subplot(2, 1, i+1)
    plt.plot(t, h, color[i], label=label[i])
    plt.axis([t1, t2, -3.5, 3.5])
    plt.legend()
    plt.ylabel('wave amplitude', fontsize=14)

plt.xlabel('time since 2015/9/14/9:50:45.44  / sec', fontsize=14)
plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
