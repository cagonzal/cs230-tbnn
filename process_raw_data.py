import numpy as np 

def integrity_basis(shat, rhat):

    '''
    Computes integrity basis tensors
    Inputs: normalized strain rate tensor, normalized rotation rate tensor
        normalization - k/2/eps

    '''

    basis = np.empty([10, 3, 3])
    eye = np.ones([3,3])

    t1  = shat
    t2  = shat @ rhat - rhat @ shat
    t3  = shat @ shat - 1/3 * np.trace(shat @ shat) * eye
    t4  = rhat @ rhat - 1/3 * np.trace(rhat @ rhat) * eye
    t5  = rhat @ shat @ shat - shat @ shat @ rhat
    t6  = rhat @ rhat @ shat + shat @ rhat @ rhat- 2/3 * np.trace(shat @ rhat @ rhat) * eye
    t7  = rhat @ shat @ rhat @ rhat - rhat @ rhat @ shat @ rhat 
    t8  = shat @ rhat @ shat @ shat - shat @ shat @ rhat @ shat
    t9  = rhat @ rhat @ shat @ shat + shat @ shat @ rhat @ rhat - 2/3 * np.trace(shat @ shat @ rhat @ rhat) * eye
    t10 = rhat @ shat @ shat @ rhat @ rhat - rhat @ rhat @ shat @ shat @ rhat 

    basis[0,:,:] = t1
    basis[1,:,:] = t2
    basis[2,:,:] = t3
    basis[3,:,:] = t4
    basis[4,:,:] = t5
    basis[5,:,:] = t6
    basis[6,:,:] = t7
    basis[7,:,:] = t8
    basis[8,:,:] = t9
    basis[9,:,:] = t10

    return basis


def compute_invariants(shat,rhat):

    '''
    Computes scalar invariants
    Inputs: normalized strain rate tensor, normalized rotation rate tensor
        normalization - k/2/eps

    '''


    lam = np.empty([1,7])
    lam[0,0] = np.trace(shat)
    lam[0,1] = np.trace(shat @ shat)
    lam[0,2] = np.trace(shat @ shat @shat)
    lam[0,3] = np.trace(rhat @ rhat)
    lam[0,4] = np.trace(rhat @ rhat @ shat)
    lam[0,5] = np.trace(rhat @ rhat @ shat @ shat)
    lam[0,6] = np.trace(rhat @ rhat @ shat @ rhat @ shat @ shat)

    return lam 


def compute_rate_tensors(gradu, Ny):
    # integrity basis tensors and scalar invariants
    sij = gradu + np.transpose(gradu, (0,2,1))
    oij = gradu - np.transpose(gradu, (0,2,1))

    return sij, oij

def compute_dissipation(sij, nu, Ny):

    eps = np.empty([Ny,1])
    for ii in range(Ny):
        eps[ii,0] = -2 * nu * np.tensordot(sij[ii,:,:], sij[ii,:,:])

    return eps

def normalize_rate_tensors(sij, oij, tke, eps, Ny):

    buf = tke / (2 * eps)

    shat = np.transpose(sij,(1,2,0)) * buf[:,0]
    shat = np.transpose(shat, (2,0,1))

    rhat = np.transpose(oij,(1,2,0)) * buf[:,0]
    rhat = np.transpose(rhat, (2,0,1))

    return shat, rhat

    
def compute_gradu(meanData, Ny):

    # mean velocity gradient tensor
    gradu = np.empty([Ny, 3, 3])
    for ii in range(Ny):
        buf = np.zeros([3,3])
        buf[0,1] = meanData[ii,3]
        gradu[ii, :, :] = buf

    return gradu


def compute_tke(flucData, Ny):

    tke = np.empty([Ny,1])
    for ii in range(Ny):
        tke[ii,0] = flucData[ii,8]

    return tke


def compute_nut(aij, sij, Ny):
    # eddy viscosity nu_t
    nut = np.empty([Ny,1])
    for ii in range(Ny):
        nut[ii,0] = -aij[ii,0,1] / (2 * sij[ii,0,1])

    return nut



def compute_bij(flucData, Ny):
    # normalized reynolds stress anisotropy tensor
    aij = np.empty([Ny, 3, 3])
    bij = np.empty([Ny, 3, 3])
    
    for ii in range(Ny):
        buf = np.array([[flucData[ii,2], flucData[ii,5], flucData[ii,6]],\
                        [flucData[ii,5], flucData[ii,3], flucData[ii,7]],\
                        [flucData[ii,6], flucData[ii,7], flucData[ii,4]] ])
        tke_temp = flucData[ii,8]
        k_dij = np.ones([3,3]) * tke_temp

        aij[ii,:,:] = buf - 2/3 * k_dij
        bij[ii,:,:] = aij[ii,:,:] / (2 * tke_temp)

    return aij, bij




def compute_qoi(shat, rhat, Ny):

    '''
    computes direct inputs to tbnn model
    inputs: ?
    outputs: lam - scalar invariants
             basis - tensor basis
    '''

    #shat, rhat = normalize_rate_tensors(sij, oij, tke, eps, Ny)

    basis = np.empty([Ny, 10, 3, 3])
    lam = np.empty([Ny,7])
    for i in range(Ny):

        basis[i,:,:,:] = integrity_basis(shat[i,:,:], rhat[i,:,:])
        lam[i,:] = compute_invariants(shat[i,:,:], rhat[i,:,:])

 
    return lam, basis




