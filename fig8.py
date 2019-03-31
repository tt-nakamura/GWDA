# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from numpy.fft import rfft, irfft, ifft, rfftfreq
from scipy.signal import butter, filtfilt
import h5py

data = [np.loadtxt('H-H1_LOSC_4_V2-1126259446-32.txt'),
        np.loadtxt('L-L1_LOSC_4_V2-1126259446-32.txt') + 1.e-18]
label = ['Hanford', 'Livingston']
color = ['b', 'g']

fs = 4096
(f_min, f_max) = (43, 300)
(t0,t1,t2) = (16.44, -0.15, 0.05)

N = len(data[0])
freq = rfftfreq(N, 1/fs)
time = np.arange(N)/fs - t0
t = time[(time >= t1) & (time < t2)]

window = np.hanning(N)

template = h5py.File('LOSC_Event_tutorial/GW150914_4_template.hdf5', 'r')
(hp,hc) = template['template']
hp = rfft(np.roll(hp[::-1]*window, N//2))
hc = rfft(np.roll(hc[::-1]*window, N//2))
template =  hc + hp*1j

(b,a) = butter(4, [f_min*2/fs, f_max*2/fs], btype='band')

plt.figure(figsize=(6.4, 5))

for (i,h) in enumerate(data):
    (S,f) = psd(h, Fs=fs, NFFT=N//8)
    S = np.interp(freq, f, S*fs)
    h = rfft(h)

    C = 2 * ifft(h * template / S, N)
    C = C[(time >= t1) & (time < t2)]
    sigma = np.sqrt(np.sum(np.abs(template)**2 / S) / N)
    rho = np.abs(C) / sigma
    r_eff = sigma / np.max(rho)
    phi = np.angle(C[np.argmax(rho)])

    h1 = (hc * np.cos(phi) + hp * np.sin(phi)) / r_eff
    h1 = irfft(h1/np.sqrt(S/2))
    h1 = filtfilt(b,a,h1)
    h1 = np.roll(h1[::-1], int((t[np.argmax(rho)] + t0)*fs))
    h1 = h1[(time >= t1) & (time < t2)]

    h = irfft(h/np.sqrt(S/2))
    h = filtfilt(b,a,h)
    h = h[(time >= t1) & (time < t2)]

    plt.subplot(2,1,i+1)
    plt.axis([t1, t2, -3.5, 3.5])
    plt.plot(t, h, color[i], label='signal')
    plt.plot(t, h1, '--k', label='template', lw=1)
    plt.text(-0.146, 2.8, label[i])
    plt.ylabel('wave amplitude', fontsize=14)
    plt.legend()

plt.xlabel('time since 2015/9/14/9:50:45.44  / sec', fontsize=14)
plt.tight_layout()
plt.savefig('fig8.eps')
plt.show()
