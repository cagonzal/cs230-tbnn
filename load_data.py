import numpy as np 
import matplotlib.pyplot as plt

def load_data(filepath,ny,ncols):
    count = 0
    data = np.empty([ny, ncols])
    with open(filepath) as fp:
        for line in fp:
            if line[0] != "%":
                proc_line = (line.strip()).split()
                data[count,:] = np.asarray(proc_line,dtype=np.float)
                count += 1

    return data

def integrity_basis(shat, rhat):

    basis = np.empty([10, 3, 3])
    eye = np.ones([3,3])

    T1 = shat
    T2 = shat @ rhat - rhat @ shat
    T3 = shat @ shat - 1/3 * np.trace(shat @ shat) * eye
    T4 = rhat @ rhat - 1/3 * np.trace(rhat @ rhat) * eye
    T5 = rhat @ shat @ shat - shat @ shat @ rhat
    T6 = rhat @ rhat @ shat + shat @ rhat @ rhat- 2/3 * np.trace(shat @ rhat @ rhat) * eye
    T7 = rhat @ shat @ rhat @ rhat - rhat @ rhat @ shat @ rhat 
    T8 = shat @ rhat @ shat @ shat - shat @ shat @ rhat @ shat
    T9 = rhat @ rhat @ shat @ shat + shat @ shat @ rhat @ rhat - 2/3 * np.trace(shat @ shat @ rhat @ rhat) * eye
    T10 = rhat @ shat @ shat @ rhat @ rhat - rhat @ rhat @ shat @ shat @ rhat 

    basis[0,:,:] = T1
    basis[1,:,:] = T2
    basis[2,:,:] = T3
    basis[3,:,:] = T4
    basis[4,:,:] = T5
    basis[5,:,:] = T6
    basis[6,:,:] = T7
    basis[7,:,:] = T8
    basis[8,:,:] = T9
    basis[9,:,:] = T10

    return basis

def compute_invariants(shat,rhat):

    lam = np.empty([1,7])
    lam[0,0] = np.trace(shat)
    lam[0,1] = np.trace(shat @ shat)
    lam[0,2] = np.trace(shat @ shat @shat)
    lam[0,3] = np.trace(rhat @ rhat)
    lam[0,4] = np.trace(rhat @ rhat @ shat)
    lam[0,5] = np.trace(rhat @ rhat @ shat @ shat)
    lam[0,6] = np.trace(rhat @ rhat @ shat @ rhat @ shat @ shat)

    return lam 

def compute_qoi(meanData, flucData, nu):

    # mean velocity gradient tensor
    gradu = np.empty([ny, 3, 3])
    for ii in range(ny):
        buff = np.zeros([3,3])
        buff[0,1] = meanData[ii,3]
        gradu[ii, :, :] = buff

    # normalized Reynolds stress anisotropy tensor
    bij = np.empty([ny, 3, 3])
    aij = np.empty([ny, 3, 3])
    for ii in range(ny):
        buff = np.array([[flucData[ii,2], flucData[ii,5], flucData[ii,6]],\
                         [flucData[ii,5], flucData[ii,3], flucData[ii,7]],\
                         [flucData[ii,6], flucData[ii,7], flucData[ii,4]] ])
        k = flucData[ii,8]
        k_dij = np.ones([3,3]) * k
        aij[ii,:,:] = buff - 2/3 * k_dij

        bij[ii,:,:] = aij[ii,:,:] / (2 * k)

    # tke
    k = np.empty([ny, 1])
    for ii in range(ny):
        k[ii,0] = flucData[ii,8]

    # integrity basis tensors and scalar invariants
    sij = gradu + np.transpose(gradu, (0,2,1))
    omij = gradu - np.transpose(gradu, (0,2,1))

    epsilon = np.empty([ny,1])
    for i in range(ny):
        epsilon[i,0] = -2 * nu * np.tensordot(sij[i,:,:], sij[i,:,:])

    buff = k / (2 * epsilon)
    shat = np.transpose(sij,(1,2,0)) * buff[:,0]
    shat = np.transpose(shat, (2,0,1))

    rhat = np.transpose(omij,(1,2,0)) * buff[:,0]
    rhat = np.transpose(rhat, (2,0,1))

    basis = np.empty([ny, 10, 3, 3])
    lam = np.empty([ny,7])
    for i in range(ny):

        basis[i,:,:,:] = integrity_basis(shat[i,:,:], rhat[i,:,:])
        lam[i,:] = compute_invariants(shat[i,:,:], rhat[i,:,:])

    # eddy viscosity nu_t
    nut = np.empty([ny,1])
    for ii in range(ny):
        nut[ii,0] = -aij[ii,0,1] / (2 * sij[ii,0,1])

    return lam, basis, gradu, bij, k, epsilon, nut



filepath_mean = 'data/re550/LM_Channel_0550_mean_prof.dat'
filepath_fluc = 'data/re550/LM_Channel_0550_vel_fluc_prof.dat'

ny = 192
nu = 1 * 10**(-4) # from the data file

# y/delta, y+, U, dU/dy, W, P
meanData = load_data(filepath_mean, ny, 6)

# y/delta, y+, u'u', v'v', w'w', u'v', u'w', v'w', k
flucData = load_data(filepath_fluc, ny, 9)

lam, tb, gradu, bij, k, epsilon, nut = compute_qoi(meanData, flucData, nu)







