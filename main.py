import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data
import process_raw_data as pd

filepath_mean = 'data/re550/LM_Channel_0550_mean_prof.dat'
filepath_fluc = 'data/re550/LM_Channel_0550_vel_fluc_prof.dat'

Ny = 192
nu = 1 * 10**(-4) # from the data file

# y/delta, y+, U, dU/dy, W, P
meanData = load_data(filepath_mean, Ny, 6)

# y/delta, y+, u'u', v'v', w'w', u'v', u'w', v'w', k
flucData = load_data(filepath_fluc, Ny, 9)

gradu = pd.compute_gradu(meanData, Ny)
aij, bij = pd.compute_bij(flucData, Ny)
tke = pd.compute_tke(flucData, Ny)

sij, oij = pd.compute_rate_tensors(gradu, Ny)
eps = pd.compute_dissipation(sij, nu, Ny)
nut = pd.compute_nut(aij, sij, Ny)

shat, rhat = pd.normalize_rate_tensors(sij, oij, tke, eps, Ny)

lam, tb = pd.compute_qoi(shat, rhat, Ny)


