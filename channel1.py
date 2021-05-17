import numpy as np
import matplotlib.pyplot as plt
#from tbnns.tbnn import TBNN
from tbnns import printInfo
import random
import sys

import load_data as ld
import process_raw_data as pd
import apply_tbnn as apptb

# Input/Settings
random.seed(10)
fsize = 3

Ny = 192
#Ny = 256
nu = 1 * 10**(-4) # from the data file

# Filenames
filepath_mean = 'data/re0550/LM_Channel_0550_mean_prof.dat'
filepath_fluc = 'data/re0550/LM_Channel_0550_vel_fluc_prof.dat'
filepath_tke  = 'data/re0550/LM_Channel_0550_RSTE_k_prof.dat'

# Load data
# y/delta, y+, U, dU/dy, W, P
y_raw, U_raw, dUdy_raw = ld.load_mean_data(filepath_mean, Ny)
# y/delta, y+, u'u', v'v', w'w', u'v', u'w', v'w', k
uus_raw, tke_raw = ld.load_fluc_data(filepath_fluc, Ny)
# y/delta, y+, ...
eps_raw = ld.load_tke_data(filepath_tke, Ny)

# Filter for synthetic RANS
y_filt    = pd.rans_filter(y_raw,    fsize)
U_filt    = pd.rans_filter(U_raw,    fsize)
dUdy_filt = pd.rans_filter(dUdy_raw, fsize)
uus_filt  = pd.rans_filter(uus_raw,  fsize)
tke_filt  = pd.rans_filter(tke_raw,  fsize)
eps_filt  = pd.rans_filter(eps_raw,  fsize)

# Shuffle data
shuffler = np.random.permutation(Ny)

y_raw_sh    = y_raw[shuffler]
U_raw_sh    = U_raw[shuffler]
dUdy_raw_sh = dUdy_raw[shuffler]
uus_raw_sh  = uus_raw[shuffler,:]
tke_raw_sh  = tke_raw[shuffler]
eps_raw_sh  = eps_raw[shuffler]

y_filt_sh    = y_filt[shuffler]
U_filt_sh    = U_filt[shuffler]
dUdy_filt_sh = dUdy_filt[shuffler]
uus_filt_sh  = uus_filt[shuffler,:]
tke_filt_sh  = tke_filt[shuffler]
eps_filt_sh  = eps_filt[shuffler]

print('Data is shuffled')

# Split into train/dev/test
ind_train = int(np.floor(0.8 * Ny))
ind_dev = int(np.floor(0.9 * Ny))

y_train    = y_filt_sh[0:ind_train]
U_train    = U_filt_sh[0:ind_train]
dUdy_train = dUdy_filt_sh[0:ind_train]
uus_train  = uus_filt_sh[0:ind_train,:]
tke_train  = tke_filt_sh[0:ind_train]
eps_train  = eps_filt_sh[0:ind_train]

y_dev    = y_raw_sh[ind_train:ind_dev]
U_dev    = U_raw_sh[ind_train:ind_dev]
dUdy_dev = dUdy_raw_sh[ind_train:ind_dev]
uus_dev  = uus_raw_sh[ind_train:ind_dev,:]
tke_dev  = tke_raw_sh[ind_train:ind_dev]
eps_dev  = eps_raw_sh[ind_train:ind_dev]

y_test    = y_raw_sh[ind_dev:]
U_test    = U_raw_sh[ind_dev:]
dUdy_test = dUdy_raw_sh[ind_dev:]
uus_test  = uus_raw_sh[ind_dev:,:]
tke_test  = tke_raw_sh[ind_dev:]
eps_test  = eps_raw_sh[ind_dev:]

Ntrain = U_train.shape[0]
Ndev   = U_dev.shape[0]
Ntest  = U_test.shape[0]


# Process data
#vol = pd.compute_cell_volumes(y, ny)

# Velocity gradient
gradu_train = pd.compute_gradu(dUdy_train, Ntrain)
gradu_dev   = pd.compute_gradu(  dUdy_dev,   Ndev)
gradu_test  = pd.compute_gradu( dUdy_test,  Ntest)
gradu_raw   = pd.compute_gradu(  dUdy_raw,     Ny)

# Rate tensors
sij_train, oij_train = pd.compute_rate_tensors(gradu_train, Ntrain)
sij_dev, oij_dev     = pd.compute_rate_tensors(  gradu_dev,   Ndev)
sij_test, oij_test   = pd.compute_rate_tensors( gradu_test,  Ntest)
sij_raw, oij_raw     = pd.compute_rate_tensors(  gradu_raw,     Ny)

# Dissipation
#eps_train = pd.compute_dissipation(sij_train, nu, Ntrain)
#eps_dev   = pd.compute_dissipation(  sij_dev, nu,   Ndev)
#eps_test  = pd.compute_dissipation( sij_test, nu,  Ntest)
#eps_raw   = pd.compute_dissipation(  sij_raw, nu,     Ny)

# Anisotropy tensor
aij_train, bij_train = pd.compute_bij(uus_train, tke_train, Ntrain)
aij_dev, bij_dev     = pd.compute_bij(  uus_dev,   tke_dev,   Ndev)
aij_test, bij_test   = pd.compute_bij( uus_test,  tke_test,  Ntest)
aij_raw, bij_raw     = pd.compute_bij(   uus_raw,  tke_raw,     Ny)

# Eddy viscosity
nut_train = pd.compute_nut(aij_train, sij_train, Ntrain)
nut_dev   = pd.compute_nut(  aij_dev,   sij_dev,   Ndev)
nut_test  = pd.compute_nut( aij_test,  sij_test,  Ntest)


# Normalize rate tensors
shat_train, rhat_train = pd.normalize_rate_tensors(sij_train, aij_train, tke_train, eps_train, Ntrain)
shat_dev, rhat_dev     = pd.normalize_rate_tensors(  sij_dev,   aij_dev,   tke_dev,   eps_dev,   Ndev)
shat_test, rhat_test   = pd.normalize_rate_tensors( sij_test,  aij_test,  tke_test,  eps_test,  Ntest)

shat_train = sij_train
rhat_train = oij_train
shat_dev = sij_dev
rhat_dev = oij_dev
shat_test = sij_test
rhat_test = oij_test

# Compute QoIs: lam = scalar invariant, tb = tensor basis
lam_train, tb_train = pd.compute_qoi(shat_train, rhat_train, Ntrain)
lam_dev, tb_dev     = pd.compute_qoi(  shat_dev,   rhat_dev,   Ndev)
lam_test, tb_test   = pd.compute_qoi( shat_test,  rhat_test,  Ntest)

print("lam_train has shape " + str(lam_train.shape))
print("lam_dev has shape " + str(lam_dev.shape))
print("lam_test has shape " + str(lam_test.shape))

print("tb_train has shape " + str(tb_train.shape))
print("tb_dev has shape " + str(tb_dev.shape))
print("tb_test has shape " + str(tb_test.shape))    

# save terminal output to file
#fout = open('output.txt','w')
#sys.stdout = fout
printInfo()

print("")
print("Training TBNN on baseline Re_tau=550 channel data...")
best_dev_loss, end_dev_loss, step_list, train_loss_list, dev_loss_list = apptb.trainNetwork(lam_train, tb_train, bij_train, lam_dev, tb_dev, bij_dev)
print("")
#trainNetwork(x_r2, tb_r2, b_r2, x_r1p5, tb_r1p5, b_r1p5, vol_r2, vol_r1)
# the last two arguments are optional; they consist of a weight that is applied
# to the loss function. Uncomment to apply the computational cell volume as a weight
    
# Apply the trained network
print("gradu has shape " + str(gradu_test.shape))
print("eddy_visc has shape " + str(nut_test.shape))
print("tke has shape " + str(tke_test.shape)) 

print("")
print("Applying trained TBNN on baseline Re_tau=550 channel data...")
b_pred, g = apptb.applyNetwork(lam_test, tb_test, bij_test, gradu_test, nut_test, tke_test)
#fout.close()

print('b_pred shape')
print(b_pred.shape)

step_list = np.array(step_list)

train_loss_list = np.array(train_loss_list)

# for some reason, dev_loss_list.shape = [Ndev , 4]
dev_loss_list = np.array(dev_loss_list)

plt.figure()
plt.plot(step_list, train_loss_list,    label='train')
plt.plot(step_list, dev_loss_list[:,0], label='dev')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend(loc = 'upper right')
plt.savefig('loss.png', bbox_inches='tight')


plt.figure()
plt.semilogx(y_test * 550, b_pred[:,0,1],'x', label='tbnn')
# plt.plot(y_test, bij_test[:,0,1],'+', label='test')
plt.semilogx(y_raw * 550, bij_raw[:,0,1],'-',label='truth')
# plt.plot(y_train, bij_train[:,0,1],'+',label='filter')
plt.ylabel(r'$b_{uv}$')
plt.xlabel(r'$y^+$')
plt.legend(loc='lower left')
#axes = plt.gca()
#axes.set_ylim([-2, 25])

plt.savefig('tbnn_performance.png', bbox_inches='tight')
