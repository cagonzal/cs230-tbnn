import numpy as np
#import matplotlib.pyplot as plt
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
nu = 1 * 10**(-4) # from the data file

# Filenames
filepath_mean = 'data/re550/LM_Channel_0550_mean_prof.dat'
filepath_fluc = 'data/re550/LM_Channel_0550_vel_fluc_prof.dat'

# Load data
# y/delta, y+, U, dU/dy, W, P
y_raw, U_raw, dUdy_raw = ld.load_mean_data(filepath_mean, Ny)
# y/delta, y+, u'u', v'v', w'w', u'v', u'w', v'w', k
uus_raw, tke_raw = ld.load_fluc_data(filepath_fluc, Ny)

# Filter for synthetic RANS
y_filt    = pd.rans_filter(y_raw,    fsize)
U_filt    = pd.rans_filter(U_raw,    fsize)
dUdy_filt = pd.rans_filter(dUdy_raw, fsize)
uus_filt  = pd.rans_filter(uus_raw,  fsize)
tke_filt  = pd.rans_filter(tke_raw,  fsize)

# Shuffle data
shuffler = np.random.permutation(Ny)

U_raw_sh    = U_raw[shuffler]
dUdy_raw_sh = dUdy_raw[shuffler]
uus_raw_sh  = uus_raw[shuffler,:]
tke_raw_sh  = tke_raw[shuffler]

U_filt_sh    = U_filt[shuffler]
dUdy_filt_sh = dUdy_filt[shuffler]
uus_filt_sh  = uus_filt[shuffler,:]
tke_filt_sh  = tke_filt[shuffler]

print('Data is shuffled')

# Split into train/dev/test
ind_train = int(np.floor(0.8 * Ny))
ind_dev = int(np.floor(0.9 * Ny))

U_train    = U_filt_sh[0:ind_train]
dUdy_train = dUdy_filt_sh[0:ind_train]
uus_train  = uus_filt_sh[0:ind_train,:]
tke_train  = tke_filt_sh[0:ind_train]

U_dev    = U_raw_sh[ind_train:ind_dev]
dUdy_dev = dUdy_raw_sh[ind_train:ind_dev]
uus_dev  = uus_raw_sh[ind_train:ind_dev,:]
tke_dev  = tke_raw_sh[ind_train:ind_dev]

U_test    = U_raw_sh[ind_dev:]
dUdy_test = dUdy_raw_sh[ind_dev:]
uus_test  = uus_raw_sh[ind_dev:,:]
tke_test  = tke_raw_sh[ind_dev:]

Ntrain = U_train.shape[0]
Ndev   = U_dev.shape[0]
Ntest  = U_test.shape[0]


# Process data
#vol = pd.compute_cell_volumes(y, ny)

# Velocity gradient
gradu_train = pd.compute_gradu(dUdy_train, Ntrain)
gradu_dev   = pd.compute_gradu(  dUdy_dev,   Ndev)
gradu_test  = pd.compute_gradu( dUdy_test,  Ntest)

# Rate tensors
sij_train, oij_train = pd.compute_rate_tensors(gradu_train, Ntrain)
sij_dev, oij_dev     = pd.compute_rate_tensors(  gradu_dev,   Ndev)
sij_test, oij_test   = pd.compute_rate_tensors( gradu_test,  Ntest)

# Dissipation
eps_train = pd.compute_dissipation(sij_train, nu, Ntrain)
eps_dev   = pd.compute_dissipation(  sij_dev, nu,   Ndev)
eps_test  = pd.compute_dissipation( sij_test, nu,  Ntest)

# Anisotropy tensor
aij_train, bij_train = pd.compute_bij(uus_train, tke_train, Ntrain)
aij_dev, bij_dev     = pd.compute_bij(  uus_dev,   tke_dev,   Ndev)
aij_test, bij_test   = pd.compute_bij( uus_test,  tke_test,  Ntest)

# Eddy viscosity
nut_train = pd.compute_nut(aij_train, sij_train, Ntrain)
nut_dev   = pd.compute_nut(  aij_dev,   sij_dev,   Ndev)
nut_test  = pd.compute_nut( aij_test,  sij_test,  Ntest)


# Normalize rate tensors
shat_train, rhat_train = pd.normalize_rate_tensors(sij_train, aij_train, tke_train, eps_train, Ntrain)
shat_dev, rhat_dev     = pd.normalize_rate_tensors(  sij_dev,   aij_dev,   tke_dev,   eps_dev,   Ndev)
shat_test, rhat_test   = pd.normalize_rate_tensors( sij_test,  aij_test,  tke_test,  eps_test,  Ntest)

# Compute QoIs: lam = scalar invariant, tb = tensor basis
lam_train, tb_train = pd.compute_qoi(sij_train, oij_train, Ntrain)
lam_dev, tb_dev     = pd.compute_qoi(  sij_dev,   oij_dev,   Ndev)
lam_test, tb_test   = pd.compute_qoi( sij_test,  oij_test,  Ntest)

print("lam_train has shape " + str(lam_train.shape))
print("lam_dev has shape " + str(lam_dev.shape))
print("lam_test has shape " + str(lam_test.shape))

print("tb_train has shape " + str(tb_train.shape))
print("tb_dev has shape " + str(tb_dev.shape))
print("tb_test has shape " + str(tb_test.shape))


# Copied from TBNN example

#def trainNetwork(x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
#                 loss_weight_train=None, loss_weight_dev=None):
    #"""
    #This function takes in training data and validation data (aka dev set)
    #and runs the training routine. We initialize parameters of the TBNN-s 
    #through the dictionary FLAGS and call nn.train() for training.
    #"""
        
    # Flags indicating parameters of the TBNN. This is a 
    # comprehensive list, with all flags that can be prescribed
#    FLAGS = {} # FLAGS is a dictionary with the following keys:
    
    # num features used to be 7
#    FLAGS['num_features'] = 3 # number of features to be used
#    FLAGS['num_basis'] = 10 # number of tensor basis in the expansion
#    FLAGS['num_layers'] = 8 # number of hidden layers
#    FLAGS['num_neurons'] = 30 # number of hidden units in each layer
    
#    FLAGS['learning_rate'] = 1e-3 # learning rate for SGD algorithm
#    FLAGS['num_epochs'] = 1000 # maximum number of epochs to run
#    FLAGS['early_stop_dev'] = 20 # after this many evaluations without improvement, stop training           
#    FLAGS['eval_every'] = 100 # when to evaluate losses
#    FLAGS['train_batch_size'] = 50 # number of points per batch
    
#    FLAGS['loss_type'] = 'l2' # loss type    
#    FLAGS['c_reg'] = 1e-7 # L2 regularization strength   
#    FLAGS['drop_prob'] = 0 # dropout probability at training time   
     
    # Initialize TBNN with given FLAGS
#    nn = TBNN()
#    nn.initializeGraph(FLAGS)   
    
    # Path to write TBNN metadata and parameters
#    path_params = 'checkpoints/nn_test_tbnn.ckpt'
#    path_class = 'nn_test_tbnn.pckl'
    
    # Train and save to disk
#    nn.train(path_params,
#             x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
#             train_loss_weight=loss_weight_train, dev_loss_weight=loss_weight_dev,
#             detailed_losses=True)
#    nn.saveToDisk("Testing TBNN", path_class)
  
#def applyNetwork(x_test, tb_test, b_test, gradu_test, nut_test, tke_test):
    #"""
    #This function takes in test data and applies previously trained TBNN.    
    #"""    
    
    # ----- Load meta-data and initialize parameters
    # path_class should correspond to the path on disk of an existing trained network 
#    path_class = 'nn_test_tbnn.pckl' 
#    nn = TBNN()
#    nn.loadFromDisk(path_class, verbose=True)
    
    # Apply TBNN on the test set to get losses
#    loss, loss_pred, _, _ = nn.getTotalLosses(x_test, tb_test, b_test)
#    loss_pred_rans = nn.getRansLoss(b_test, gradu_test, tke_test, nut_test)   
#    print("JICF r=1 LEVM loss: {:g}".format(loss_pred_rans))
#    print("JICF r=1 TBNN-s loss: {:g}".format(loss_pred))
#    print("")
    
    # Now, apply TBNN on the test set to get a predicted anisotropy matrix
    # This can be used in a RANS solver to get improved velocity field predictions
#    b_pred, g = nn.getTotalAnisotropy(x_test, tb_test)    
#    print("Predicted anisotropy shape: ", b_pred.shape)
#    print("Predicted coefficients g_n shape: ", g.shape)    
    

# save terminal output to file
fout = open('output.txt','w')
sys.stdout = fout
printInfo()

print("")
print("Training TBNN on baseline Re_tau=550 channel data...")
apptb.trainNetwork(lam_train, tb_train, bij_train, lam_dev, tb_dev, bij_dev)
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
apptb.applyNetwork(lam_test, tb_test, bij_test, gradu_test, nut_test, tke_test)
fout.close()

