# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from matplotlib import rc

data = [np.loadtxt('H-H1_LOSC_4_V2-1126259446-32.txt'),
        np.loadtxt('L-L1_LOSC_4_V2-1126259446-32.txt') + 1.e-18]
label = ['Hanford', 'Livingston']
color = ['b', 'g']

fs = 4096
N = len(data[0])

plt.figure(figsize=(6.4, 6))
rc('text', usetex=True)

for (i,h) in enumerate(data):
    (S,f) = psd(h, Fs=fs, NFFT=N//8)
    plt.subplot(2,1,i+1)
    plt.axis([20, 2000, 5e-23, 5e-18])
    plt.loglog(f, np.sqrt(fs*S/2), color[i], label=label[i], lw=1)
    plt.legend()
    plt.ylabel(r'$\sqrt{f_{\rm c}\, S(f)}$', fontsize=14)

plt.xlabel('frequency  / Hz', fontsize=14)
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
