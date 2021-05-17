import numpy as np 

def rans_filter(x, w):

    '''
    Moving average window
    Altered at endpoint for symmetry condition in centerline
        Ok at beginning because of no slip
        Currently only works for w = 3
    '''
    if x.ndim > 1 and x.shape[1] > 1:
        out = np.empty_like(x)
        for ii in range(x.shape[1]):
            out[:,ii] = np.convolve(x[:,ii], np.ones(w), 'same') / w
            out[-1,ii] = (x[-1,ii] + x[-2,ii] + x[-3,ii]) / w
            out[-2,ii] = (x[-2,ii] + x[-1,ii] + x[-2,ii]) / w
    else:
        out = np.convolve(x, np.ones(w), 'same') / w
        out[-1] = (x[-1] + x[-2] + x[-3]) / w
        out[-2] = (x[-2] + x[-1] + x[-2]) / w

    return out

def integrity_basis(shat, rhat):

    '''
    Computes integrity basis tensors
    Inputs: normalized strain rate tensor, normalized rotation rate tensor
        normalization - k/2/eps

    '''

    basis = np.empty([10, 3, 3])
    eye = np.identity(3)

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
 
    lam = np.empty([1,3])
    lam[0,0] = np.trace(shat @ shat)
    lam[0,1] = np.trace(rhat @ rhat)
    lam[0,2] = np.trace(rhat @ rhat @ shat @ shat)

    return lam 


def compute_rate_tensors(gradu, Ny):
    # integrity basis tensors and scalar invariants
    sij = 1/2 * ( gradu + np.transpose(gradu, (0,2,1)) )
    oij = 1/2 * ( gradu - np.transpose(gradu, (0,2,1)) )

    return sij, oij


def normalize_rate_tensors(sij, oij, tke, eps, Ny):

    buf = tke / (2 * eps)

    shat = np.transpose(sij,(1,2,0)) * buf
    shat = np.transpose(shat, (2,0,1))

    rhat = np.transpose(oij,(1,2,0)) * buf
    rhat = np.transpose(rhat, (2,0,1))

    return shat, rhat

    
def compute_gradu(dUdy, Ny):

    # mean velocity gradient tensor
    gradu = np.empty([Ny, 3, 3])
    for ii in range(Ny):
        buf = np.zeros([3,3])
        buf[0,1] = dUdy[ii]
        gradu[ii, :, :] = buf

    return gradu


def compute_cell_volumes(y, Ny):

    # outputs volumes in as V/delta^3
    # convert to plus units by multiplying by Re^3

    dx = 8.*np.pi / 1536
    dz = 3.*np.pi / 1024
    cellVols = np.empty([Ny])

    cellVols[0] = 0.

    for ii in range(1,Ny-1):
        dy = (y[ii+1]-y[ii-1])/2
        cellVols[ii] = dy * dx*dz

    dy = y[-1]-y[-2]
    cellVols[-1] = dy * dx*dz

    return cellVols


def compute_nut(aij, sij, Ny):
    # eddy viscosity nu_t
    nut = np.empty([Ny])
    for ii in range(Ny):
        nut[ii] = -aij[ii,0,1] / (2 * sij[ii,0,1])

    return nut



def compute_bij(uus, tke, Ny):
    # normalized reynolds stress anisotropy tensor
    aij = np.empty([Ny, 3, 3])
    bij = np.empty([Ny, 3, 3])
    
    for ii in range(Ny):
        buf = np.array([[uus[ii,0], uus[ii,3], uus[ii,4]],\
                        [uus[ii,3], uus[ii,1], uus[ii,5]],\
                        [uus[ii,4], uus[ii,5], uus[ii,2]] ])
        k_dij = np.identity(3) * tke[ii]

        aij[ii,:,:] = buf - 2/3 * k_dij
        bij[ii,:,:] = aij[ii,:,:] / (2 * tke[ii])

    return aij, bij



def compute_qoi(shat, rhat, Ny):

    '''
    computes direct inputs to tbnn model
    outputs: lam - scalar invariants
             basis - tensor basis
    '''
 
    basis = np.empty([Ny, 10, 3, 3])
    lam = np.empty([Ny,3])
    for i in range(Ny):

        basis[i,:,:,:] = integrity_basis(shat[i,:,:], rhat[i,:,:])
        lam[i,:] = compute_invariants(shat[i,:,:], rhat[i,:,:])

 
    return lam, basis


