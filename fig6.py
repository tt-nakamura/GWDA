# data from:
#  LIGO open science center
#   www.gw-openscience.org/GW150914data/LOSC_Event_tutorial_GW150914.html

import numpy as np
import matplotlib.pyplot as plt
import h5py

template = h5py.File('LOSC_Event_tutorial/GW150914_4_template.hdf5', 'r')

hp, hc = template["template"]
N = len(hp)//2
hp = hp[N:0:-1]
hc = hc[N:0:-1]

t = np.linspace(0, 16, N, endpoint=False)

plt.figure(figsize=(6.4, 3.2))

plt.plot(t, hc, 'r', label='cosine mode', lw=1)
plt.plot(t, hp, 'm--', label='sine mode', lw=1)
plt.legend()
plt.xlim([0,1])
plt.xlabel('time until coalescence  / sec', fontsize=14)
plt.ylabel('waveform template', fontsize=14)
plt.yticks(np.arange(-1e-18, 1.1e-18, step=5e-19))
plt.tight_layout()
plt.savefig('fig6.eps')
plt.show()
