# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt

data = [np.loadtxt('H-H1_LOSC_4_V2-1126259446-32.txt'),
        np.loadtxt('L-L1_LOSC_4_V2-1126259446-32.txt') + 1.e-18]
label = ['Hanford', 'Livingston']
color = ['b', 'g']

(t0,t1,t2) = (16.44, -5, 5)

N = len(data[0])
time = np.linspace(0, 32, N, endpoint=False) - t0
t = time[(time>=t1) & (time<t2)]

plt.figure(figsize=(6.4, 5))

for (i,h) in enumerate(data):
    h = h[(time >= t1) & (time < t2)]
    plt.subplot(2, 1, i+1)
    plt.axis([t1, t2, -8e-19, 8e-19])
    plt.plot(t, h, color[i], label=label[i], lw=1)
    plt.legend(loc='upper right')
    plt.ylabel('detector output', fontsize=14)

plt.xlabel('time since 2015/9/14/9:50:45.44  / sec', fontsize=14)
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
