import numpy as np
import matplotlib.pyplot as plt
from tbnns import printInfo
import random
import sys

import load_data as ld
import process_raw_data as pd
import apply_tbnn as apptb

# Input/Settings
seed_no = 7
np.random.seed(seed_no)
fsize = 3

Ny = 192
Re = 550
Ny_test = 256
Re_test = 1000

# Filenames
filepath_mean = 'data/channel/re0550/LM_Channel_0550_mean_prof.dat'
filepath_fluc = 'data/channel/re0550/LM_Channel_0550_vel_fluc_prof.dat'
filepath_tke  = 'data/channel/re0550/LM_Channel_0550_RSTE_k_prof.dat'

filepath_mean_test = 'data/channel/re1000/LM_Channel_1000_mean_prof.dat'
filepath_fluc_test = 'data/channel/re1000/LM_Channel_1000_vel_fluc_prof.dat'
filepath_tke_test  = 'data/channel/re1000/LM_Channel_1000_RSTE_k_prof.dat'


# Load data
y_train, U_train, dUdy_train = ld.load_mean_data(filepath_mean, Ny)
uus_train, tke_train = ld.load_fluc_data(filepath_fluc, Ny)
eps_train = ld.load_tke_data(filepath_tke, Ny)

y_raw, U_raw, dUdy_raw = ld.load_mean_data(filepath_mean_test, Ny_test)
uus_raw, tke_raw = ld.load_fluc_data(filepath_fluc_test, Ny_test)
eps_raw = ld.load_tke_data(filepath_tke_test, Ny_test)


# Filter for synthetic RANS
y_filt_train    = pd.rans_filter(y_train,    fsize)
U_filt_train    = pd.rans_filter(U_train,    fsize)
dUdy_filt_train = pd.rans_filter(dUdy_train, fsize)
uus_filt_train  = pd.rans_filter(uus_train,  fsize)
tke_filt_train  = pd.rans_filter(tke_train,  fsize)
eps_filt_train  = pd.rans_filter(eps_train,  fsize)

y_filt    = pd.rans_filter(y_raw,    fsize)
U_filt    = pd.rans_filter(U_raw,    fsize)
dUdy_filt = pd.rans_filter(dUdy_raw, fsize)
uus_filt  = pd.rans_filter(uus_raw,  fsize)
tke_filt  = pd.rans_filter(tke_raw,  fsize)
eps_filt  = pd.rans_filter(eps_raw,  fsize)


# Shuffle data
shuffler = np.random.permutation(Ny)
y_train    = y_filt_train[shuffler]
U_train    = U_filt_train[shuffler]
dUdy_train = dUdy_filt_train[shuffler]
uus_train  = uus_filt_train[shuffler,:]
tke_train  = tke_filt_train[shuffler]
eps_train  = eps_filt_train[shuffler]

shuffler = np.random.permutation(Ny_test)
y_filt_sh    = y_filt[shuffler]
U_filt_sh    = U_filt[shuffler]
dUdy_filt_sh = dUdy_filt[shuffler]
uus_filt_sh  = uus_filt[shuffler,:]
tke_filt_sh  = tke_filt[shuffler]
eps_filt_sh  = eps_filt[shuffler]


# Split into train/dev/test
ind_dev = int(np.floor(0.5 * Ny_test))

y_dev    = y_filt_sh[0:ind_dev]
U_dev    = U_filt_sh[0:ind_dev]
dUdy_dev = dUdy_filt_sh[0:ind_dev]
uus_dev  = uus_filt_sh[0:ind_dev]
tke_dev  = tke_filt_sh[0:ind_dev]
eps_dev  = eps_filt_sh[0:ind_dev]

y_test    = y_filt_sh[ind_dev:]
U_test    = U_filt_sh[ind_dev:]
dUdy_test = dUdy_filt_sh[ind_dev:]
uus_test  = uus_filt_sh[ind_dev:,:]
tke_test  = tke_filt_sh[ind_dev:]
eps_test  = eps_filt_sh[ind_dev:]


Ntrain = U_train.shape[0]
Ndev   = U_dev.shape[0]
Ntest  = U_test.shape[0]


# Calculate cell volumes (optional)
#vol = pd.compute_cell_volumes(y, ny)

# Velocity gradient
gradu_train = pd.compute_gradu(dUdy_train, Ntrain)
gradu_dev   = pd.compute_gradu(  dUdy_dev,   Ndev)
gradu_test  = pd.compute_gradu( dUdy_test,  Ntest)

# Rate tensors
sij_train, oij_train = pd.compute_rate_tensors(gradu_train, Ntrain)
sij_dev, oij_dev     = pd.compute_rate_tensors(  gradu_dev,   Ndev)
sij_test, oij_test   = pd.compute_rate_tensors( gradu_test,  Ntest)

# Normalize rate tensors
sij_train, oij_train = pd.normalize_rate_tensors(sij_train, oij_train, tke_train, eps_train, Ny)
sij_dev, oij_dev = pd.normalize_rate_tensors(sij_dev, oij_dev, tke_dev, eps_dev, Ny)
sij_test, oij_test = pd.normalize_rate_tensors(sij_test, oij_test, tke_test, eps_test, Ny)

# Anisotropy tensor
aij_train, bij_train = pd.compute_bij(uus_train, tke_train,  Ntrain)
aij_dev, bij_dev     = pd.compute_bij(  uus_dev,   tke_dev,    Ndev)
aij_test, bij_test   = pd.compute_bij( uus_test,  tke_test,   Ntest)
_, bij_raw           = pd.compute_bij(  uus_raw,   tke_raw, Ny_test)

# Eddy viscosity
nut_train = pd.compute_nut(aij_train, sij_train, Ntrain)
nut_dev   = pd.compute_nut(  aij_dev,   sij_dev,   Ndev)
nut_test  = pd.compute_nut( aij_test,  sij_test,  Ntest)

# Compute QoIs: lam = scalar invariant, tb = tensor basis
lam_train, tb_train = pd.compute_qoi(sij_train, oij_train, Ntrain)
lam_dev, tb_dev     = pd.compute_qoi(  sij_dev,   oij_dev,   Ndev)
lam_test, tb_test   = pd.compute_qoi( sij_test,  oij_test,  Ntest)

# save terminal output to file
fout = open('logs/channel3.txt','w')
sys.stdout = fout
printInfo()

# Train network
print("")
print("Training TBNN on baseline Re_tau=550 channel data...")
best_dev_loss, end_dev_loss, step_list, train_loss_list, dev_loss_list = apptb.trainNetwork(lam_train, tb_train, bij_train, lam_dev, tb_dev, bij_dev)
print("")
    
# Apply the trained network
print("gradu has shape " + str(gradu_test.shape))
print("eddy_visc has shape " + str(nut_test.shape))
print("tke has shape " + str(tke_test.shape)) 

print("")
print("Applying trained TBNN on baseline Re_tau=550 channel data...")
b_pred, g = apptb.applyNetwork(lam_test, tb_test, bij_test, gradu_test, nut_test, tke_test)
fout.close()

# Loss variables
step_list = np.array(step_list)
train_loss_list = np.array(train_loss_list)
# for some reason, dev_loss_list.shape = [Ndev , 4]
dev_loss_list = np.array(dev_loss_list)

# Plot
plt.figure()
plt.plot(step_list, train_loss_list,    label='Train')
plt.plot(step_list, dev_loss_list[:,0], label='Dev')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.savefig(f'figs/channel/loss4_{seed_no}.png', bbox_inches='tight')


plt.figure()
plt.semilogx(y_test * Re_test, b_pred[:,0,1],'x', label='TBNN')
plt.semilogx(y_raw * Re, bij_raw[:,0,1],'-',label='DNS')
plt.ylabel(r'$b_{12}$')
plt.xlabel(r'$y^+$')
plt.legend(loc='lower left')
plt.savefig(f'figs/channel/tbnn4_log_{seed_no}.png', bbox_inches='tight')

plt.figure()
plt.plot(y_test, b_pred[:,0,1],'x', label='TBNN')
plt.plot(y_raw, bij_raw[:,0,1],'-',label='DNS')
plt.ylabel(r'$b_{12}$')
plt.xlabel(r'$y$')
plt.legend(loc='lower right')
plt.savefig(f'figs/channel/tbnn4_linear_{seed_no}.png', bbox_inches='tight')
