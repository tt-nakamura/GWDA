# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from numpy.fft import rfft, ifft, rfftfreq
import h5py

data = [np.loadtxt('H-H1_LOSC_4_V2-1126259446-32.txt'),
        np.loadtxt('L-L1_LOSC_4_V2-1126259446-32.txt') + 1.e-18]
label = ['Hanford', 'Livingston']
color = ['b', 'g']

(t0,t1,t2) = (16.44, -0.15, 0.05)

fs = 4096
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

plt.figure(figsize=(6.4, 5.4))

for (i,h) in enumerate(data):
    (S,f) = psd(h, Fs=fs, NFFT=N//8)
    S = np.interp(freq, f, S*fs)
    h = rfft(h * window)

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

    plt.subplot(2,1,i+1)
    plt.axis([t1, t2, 0, 22])
    plt.plot(t, rho, color[i], label=label[i])
    plt.ylabel('signal to noise ratio', fontsize=14)
    plt.legend(loc='upper left')

plt.xlabel('time since 2015/9/14/9:50:45.44  / sec', fontsize=14)
plt.tight_layout()
plt.savefig('fig7.eps')
plt.show()
