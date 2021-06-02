import numpy as np
import matplotlib.pyplot as plt
from tbnns import printInfo
import random
import sys

import load_data as ld
import process_raw_data as pd
import apply_tbnn as apptb

# Input/Settings
seed_no = 3
np.random.seed(seed_no)
fsize = 3

Re = 180

# Filenames
filepath = 'data/shs/tbnn_stats.npz'

# Load data
Ny, y_raw, U_raw, dUdy_raw, uus_raw, tke_raw, eps_raw = ld.load_shs_data(filepath)


# Filter for synthetic RANS
y_filt    = pd.rans_filter(y_raw,    fsize)
U_filt    = pd.rans_filter(U_raw,    fsize)
dUdy_filt = pd.rans_filter(dUdy_raw, fsize)
uus_filt  = pd.rans_filter(uus_raw,  fsize)
tke_filt  = pd.rans_filter(tke_raw,  fsize)
eps_filt  = pd.rans_filter(eps_raw,  fsize)


# Shuffle data
shuffler = np.random.permutation(Ny)

y_filt_sh    = y_filt[shuffler]
U_filt_sh    = U_filt[shuffler]
dUdy_filt_sh = dUdy_filt[shuffler]
uus_filt_sh  = uus_filt[shuffler,:]
tke_filt_sh  = tke_filt[shuffler]
eps_filt_sh  = eps_filt[shuffler]


# Split into train/dev/test
ind_train = int(np.floor(0.8 * Ny))
ind_dev = int(np.floor(0.9 * Ny))

y_train    = y_filt_sh[0:ind_train]
U_train    = U_filt_sh[0:ind_train]
dUdy_train = dUdy_filt_sh[0:ind_train]
uus_train  = uus_filt_sh[0:ind_train,:]
tke_train  = tke_filt_sh[0:ind_train]
eps_train  = eps_filt_sh[0:ind_train]

y_dev    = y_filt_sh[ind_train:ind_dev]
U_dev    = U_filt_sh[ind_train:ind_dev]
dUdy_dev = dUdy_filt_sh[ind_train:ind_dev]
uus_dev  = uus_filt_sh[ind_train:ind_dev,:]
tke_dev  = tke_filt_sh[ind_train:ind_dev]
eps_dev  = eps_filt_sh[ind_train:ind_dev]

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
aij_train, bij_train = pd.compute_bij(uus_train, tke_train, Ntrain)
aij_dev, bij_dev     = pd.compute_bij(  uus_dev,   tke_dev,   Ndev)
aij_test, bij_test   = pd.compute_bij( uus_test,  tke_test,  Ntest)
_, bij_raw           = pd.compute_bij(   uus_raw,  tke_raw,     Ny)

# Eddy viscosity
nut_train = pd.compute_nut(aij_train, sij_train, Ntrain)
nut_dev   = pd.compute_nut(  aij_dev,   sij_dev,   Ndev)
nut_test  = pd.compute_nut( aij_test,  sij_test,  Ntest)

# Compute QoIs: lam = scalar invariant, tb = tensor basis
lam_train, tb_train = pd.compute_qoi(sij_train, oij_train, Ntrain)
lam_dev, tb_dev     = pd.compute_qoi(  sij_dev,   oij_dev,   Ndev)
lam_test, tb_test   = pd.compute_qoi( sij_test,  oij_test,  Ntest)

# save terminal output to file
fout = open('logs/shs1.txt','w')
sys.stdout = fout
printInfo()

# Train network
print("")
print("Training TBNN on baseline Re_tau=550 channel data...")
best_dev_loss, end_dev_loss, step_list, train_loss_list, dev_loss_list = apptb.trainNetwork(lam_train, tb_train, bij_train, lam_dev, tb_dev, bij_dev)
print("")
 

# MOVED FOR DEBUGGING
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
plt.savefig(f'figs/shs/loss1_{seed_no}.png', bbox_inches='tight')
# END MOVED FOR DEBUGGING

# Apply the trained network
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
plt.legend(loc = 'lower left')
plt.savefig(f'figs/shs/loss1_{seed_no}.png', bbox_inches='tight')


plt.figure()
plt.semilogx(y_test * Re, b_pred[:,0,1],'x', label='TBNN')
plt.semilogx(y_raw * Re, bij_raw[:,0,1],'-',label='DNS')
plt.ylabel(r'$b_{12}$')
plt.xlabel(r'$y^+$')
plt.legend(loc='lower left')
plt.savefig(f'figs/shs/tbnn1_log_{seed_no}.png', bbox_inches='tight')

plt.figure()
plt.plot(y_test, b_pred[:,0,1],'x', label='TBNN')
plt.plot(y_raw, bij_raw[:,0,1],'-',label='DNS')
plt.ylabel(r'$b_{12}$')
plt.xlabel(r'$y$')
plt.legend(loc='lower right')
plt.savefig(f'figs/shs/tbnn1_linear_{seed_no}.png', bbox_inches='tight')
