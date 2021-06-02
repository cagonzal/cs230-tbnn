import numpy as np
import matplotlib.pyplot as plt
import sys

import load_data as ld

# Input/Settings
Ny1 = 192
Ny2 = 256
Ny3 = 96
Ny4 = 128

# Filenames - channel
filepath_mean1 = 'data/channel/re0550/LM_Channel_0550_mean_prof.dat'
filepath_fluc1 = 'data/channel/re0550/LM_Channel_0550_vel_fluc_prof.dat'
filepath_mean2 = 'data/channel/re1000/LM_Channel_1000_mean_prof.dat'
filepath_fluc2 = 'data/channel/re1000/LM_Channel_1000_vel_fluc_prof.dat'

# Filenames - couette
filepath_mean3 = 'data/couette/re0220/LM_Couette_R0220_100PI_mean_prof.dat'
filepath_fluc3 = 'data/couette/re0220/LM_Couette_R0220_100PI_vel_fluc_prof.dat'
filepath_mean4 = 'data/couette/re0500/LM_Couette_R0500_100PI_mean_prof.dat'
filepath_fluc4 = 'data/couette/re0500/LM_Couette_R0500_100PI_vel_fluc_prof.dat'

# Filenames - shs
filepath_shs = 'data/shs/tbnn_stats.npz'

# Load data
y1, U1 = ld.load_mean_data(filepath_mean1, Ny1)[0:2]
tke1 = ld.load_fluc_data(filepath_fluc1, Ny1)[1]

y2, U2 = ld.load_mean_data(filepath_mean2, Ny2)[0:2]
tke2 = ld.load_fluc_data(filepath_fluc2, Ny2)[1]

y3, U3 = ld.load_mean_data(filepath_mean3, Ny3)[0:2]
tke3 = ld.load_fluc_data(filepath_fluc3, Ny3)[1]

y4, U4 = ld.load_mean_data(filepath_mean4, Ny4)[0:2]
tke4 = ld.load_fluc_data(filepath_fluc4, Ny4)[1]

_, y5, U5, _, _, tke5, _ = ld.load_shs_data(filepath_shs)


# Plot
plt.figure()
plt.plot(U1, y1, '--', label='Channel, Re = 550')
plt.plot(U2, y2, '--', label='Channel, Re = 1000')
plt.plot(U3, y3, '--', label='Couette, Re = 220')
plt.plot(U4, y4, '--', label='Couette, Re = 500')
plt.plot(U5, y5, '--', label='SHS, Re = 180')
plt.xlabel('U')
plt.ylabel('y')
plt.legend()
plt.savefig(f'figs/all_vels.png', bbox_inches='tight')

plt.figure()
plt.plot(tke1, y1, '--', label='Channel, Re = 550')
plt.plot(tke2, y2, '--', label='Channel, Re = 1000')
plt.plot(tke3, y3, '--', label='Couette, Re = 220')
plt.plot(tke4, y4, '--', label='Couette, Re = 500')
plt.plot(tke5, y5, '--', label='SHS, Re = 180')
plt.xlabel('TKE')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.savefig(f'figs/all_tkes.png', bbox_inches='tight')


#plt.figure()
#plt.plot(U1, y1)
#plt.xlabel('U')
#plt.ylabel('y')
#plt.savefig(f'figs/channel/vel_0550.png', bbox_inches='tight')

#plt.figure()
#plt.plot(tke1,y1)
#plt.xlabel('TKE')
#plt.ylabel('y')
#plt.savefig(f'figs/channel/tke_0550.png', bbox_inches='tight')

