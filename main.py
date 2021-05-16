import numpy as np
#import matplotlib.pyplot as plt
from tbnns.tbnn import TBNN
from tbnns import printInfo
import random
import sys

import load_data as ld
import process_raw_data as pd

# Load data
filepath_mean = 'data/re550/LM_Channel_0550_mean_prof.dat'
filepath_fluc = 'data/re550/LM_Channel_0550_vel_fluc_prof.dat'

Ny_raw = 192
nu = 1 * 10**(-4) # from the data file

# y/delta, y+, U, dU/dy, W, P
#meanData = ld.load_data(filepath_mean, Ny, 6)
y_raw, U_raw, dUdy_raw = ld.load_mean_data(filepath_mean, Ny_raw)
print("y_raw has shape " + str(y_raw.shape))
print("U_raw has shape " + str(U_raw.shape))


# y/delta, y+, u'u', v'v', w'w', u'v', u'w', v'w', k
uus_raw, tke_raw = ld.load_fluc_data(filepath_fluc, Ny_raw)

# Temp fix for raw
Ny = Ny_raw
dUdy = dUdy_raw
uus = uus_raw

# Process data
#vol = pd.compute_cell_volumes(y, Ny)

gradu = pd.compute_gradu(dUdy, Ny)
print("gradu has shape " + str(gradu.shape))

#tke = pd.compute_tke(flucData, Ny)
tke_raw = np.squeeze(tke_raw)
tke = tke_raw
print("tke has shape " + str(tke.shape))


aij, bij = pd.compute_bij(uus, tke, Ny)
print("aij has shape " + str(aij.shape))
print("bij has shape " + str(bij.shape))

sij, oij = pd.compute_rate_tensors(gradu, Ny)
eps = pd.compute_dissipation(sij, nu, Ny)
nut = pd.compute_nut(aij, sij, Ny)

shat, rhat = pd.normalize_rate_tensors(sij, oij, tke, eps, Ny)

# lam = scalar invariants, tb = tensor basis
lam, tb = pd.compute_qoi(sij, oij, Ny)
print("lam has shape " + str(lam.shape))
print("tb has shape " + str(tb.shape))

random.seed(10)
shuffler = np.random.permutation(Ny)

lamShuffled = lam[shuffler,:]
tbShuffled = tb[shuffler,:,:,:]
bijShuffled = bij[shuffler,:,:]
graduShuffled = gradu[shuffler,:,:]
nutShuffled = nut[shuffler]
tkeShuffled = tke[shuffler]

print('Data is shuffled')

nTrain = int(np.floor(0.8 * Ny))
lamTrain = lamShuffled[0:nTrain,:]
tbTrain = tbShuffled[0:nTrain,:,:,:]
bijTrain = bijShuffled[0:nTrain,:,:]

nDev = int(np.floor(0.9 * Ny))
lamDev = lamShuffled[nTrain:nDev,:]
tbDev = tbShuffled[nTrain:nDev,:,:,:]
bijDev = bijShuffled[nTrain:nDev,:,:]

lamTest = lamShuffled[nDev:,:]
tbTest = tbShuffled[nDev:,:,:,:]
bijTest = bijShuffled[nDev:,:,:]

graduTest = graduShuffled[nDev:,:,:]
nutTest = np.squeeze(nutShuffled[nDev:,:])
tkeTest = tkeShuffled[nDev:]   

# Copied from TBNN example

def trainNetwork(x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
                 loss_weight_train=None, loss_weight_dev=None):
    """
    This function takes in training data and validation data (aka dev set)
    and runs the training routine. We initialize parameters of the TBNN-s 
    through the dictionary FLAGS and call nn.train() for training.
    """
        
    # Flags indicating parameters of the TBNN. This is a 
    # comprehensive list, with all flags that can be prescribed
    FLAGS = {} # FLAGS is a dictionary with the following keys:
    
    # num features used to be 7
    FLAGS['num_features'] = 3 # number of features to be used
    FLAGS['num_basis'] = 10 # number of tensor basis in the expansion
    FLAGS['num_layers'] = 8 # number of hidden layers
    FLAGS['num_neurons'] = 30 # number of hidden units in each layer
    
    FLAGS['learning_rate'] = 1e-3 # learning rate for SGD algorithm
    FLAGS['num_epochs'] = 1000 # maximum number of epochs to run
    FLAGS['early_stop_dev'] = 20 # after this many evaluations without improvement, stop training           
    FLAGS['eval_every'] = 100 # when to evaluate losses
    FLAGS['train_batch_size'] = 50 # number of points per batch
    
    FLAGS['loss_type'] = 'l2' # loss type    
    FLAGS['c_reg'] = 1e-7 # L2 regularization strength   
    FLAGS['drop_prob'] = 0 # dropout probability at training time   
     
    # Initialize TBNN with given FLAGS
    nn = TBNN()
    nn.initializeGraph(FLAGS)   
    
    # Path to write TBNN metadata and parameters
    path_params = 'checkpoints/nn_test_tbnn.ckpt'
    path_class = 'nn_test_tbnn.pckl'
    
    # Train and save to disk
    nn.train(path_params,
             x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
             train_loss_weight=loss_weight_train, dev_loss_weight=loss_weight_dev,
             detailed_losses=True)
    nn.saveToDisk("Testing TBNN", path_class)
  
def applyNetwork(x_test, tb_test, b_test, gradu_test, nut_test, tke_test):
    """
    This function takes in test data and applies previously trained TBNN.    
    """    
    
    # ----- Load meta-data and initialize parameters
    # path_class should correspond to the path on disk of an existing trained network 
    path_class = 'nn_test_tbnn.pckl' 
    nn = TBNN()
    nn.loadFromDisk(path_class, verbose=True)
    
    # Apply TBNN on the test set to get losses
    loss, loss_pred, _, _ = nn.getTotalLosses(x_test, tb_test, b_test)
    loss_pred_rans = nn.getRansLoss(b_test, gradu_test, tke_test, nut_test)   
    print("JICF r=1 LEVM loss: {:g}".format(loss_pred_rans))
    print("JICF r=1 TBNN-s loss: {:g}".format(loss_pred))
    print("")
    
    # Now, apply TBNN on the test set to get a predicted anisotropy matrix
    # This can be used in a RANS solver to get improved velocity field predictions
    b_pred, g = nn.getTotalAnisotropy(x_test, tb_test)    
    print("Predicted anisotropy shape: ", b_pred.shape)
    print("Predicted coefficients g_n shape: ", g.shape)    
    

#def main():       
    
    #printInfo() # simple function to print info about package
    
    # Load data to test TBNN
    #(x_r1, tb_r1, gradu_r1, b_r1, tke_r1, omg_r1, nut_r1, vol_r1,
    # x_r1p5, tb_r1p5, gradu_r1p5, b_r1p5, tke_r1p5, omg_r1p5, nut_r1p5, vol_r1p5,
    # x_r2, tb_r2, gradu_r2, b_r2, tke_r2, omg_r2, nut_r2, vol_r2) = loadData() 

    # train the TBNN on r=2 data (with r=1.5 as the dev set)

    # save terminal output to file
fout = open('output.txt','w')
sys.stdout = fout
printInfo()

print("")
print("Training TBNN on baseline Re_tau=550 channel data...")
trainNetwork(lamTrain, tbTrain, bijTrain, lamDev, tbDev, bijDev)
print("")
    #trainNetwork(x_r2, tb_r2, b_r2, x_r1p5, tb_r1p5, b_r1p5)    
    #trainNetwork(x_r2, tb_r2, b_r2, x_r1p5, tb_r1p5, b_r1p5, vol_r2, vol_r1)
    # the last two arguments are optional; they consist of a weight that is applied
    # to the loss function. Uncomment to apply the computational cell volume as a weight
    
    # Apply the trained network on the r=1 data

#TEST
print("gradu has shape " + str(graduTest.shape))
print("eddy_visc has shape " + str(nutTest.shape))
print("tke has shape " + str(tkeTest.shape)) 

print("")
print("Applying trained TBNN on baseline Re_tau=550 channel data...")
    #applyNetwork(x_r1, tb_r1, b_r1, gradu_r1, nut_r1, tke_r1)
applyNetwork(lamTest, tbTest, bijTest, graduTest, nutTest, tkeTest)
fout.close()

