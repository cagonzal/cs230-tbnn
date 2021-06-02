import numpy as np 

def load_shs_data(filepath):
    data = np.load(filepath)

    # mean data
    y = data['y']
    U = data['u1'] 
    dUdy = data['du1']

    # fluc data
    uus = np.zeros([Ny, 6])
    uus[:,1] = data['uu']
    uus[:,2] = data['vv']
    uus[:,3] = data['ww']
    uus[:,4] = data['uv']
    uus[:,5] = data['uw']
    uus[:,6] = data['vw']

    tke = data['tke']

    # tke data
    eps = data['eps']

    Ny = y.shape[0]

    return Ny, y, U, dUdy, uus, tke, eps



def load_fluc_data(filepath,Ny):
    ncols = 9
    count = 0
    data = np.empty([Ny, ncols])
    with open(filepath) as fp:
        for line in fp:
            if line[0] != "%":
                proc_line = (line.strip()).split()
                data[count,:] = np.asarray(proc_line,dtype=np.float)
                count += 1

    #y = data[:,0]
    #yp = data[:,1]
    #uu = data[:,2]
    #vv = data[:,3]
    #ww = data[:,4]
    #uv = data[:,5]
    #uw = data[:,6]
    #vw = data[:,7]

    uus = data[:,2:8]
    tke = data[:,8]

    return uus, tke



def load_mean_data(filepath, Ny):
    ncols = 6
    count = 0
    data = np.empty([Ny, ncols])

    with open(filepath) as fp:
        for line in fp:
            if line[0] != "%":
                proc_line = (line.strip()).split()
                data[count,:] = np.asarray(proc_line,dtype=np.float)
                count += 1

    y = data[:,0]
    #yp = data[:,1]
    U = data[:,2]
    dUdy = data[:,3]
    #W = data[:,4]
    #P = data[:,5]

    return y, U, dUdy


def load_tke_data(filepath, Ny):
    ncols = 9
    count = 0
    data = np.empty([Ny, ncols])

    with open(filepath) as fp:
        for line in fp:
            if line[0] != "%":
                proc_line = (line.strip()).split()
                data[count,:] = np.asarray(proc_line,dtype=np.float)
                count += 1

    y = data[:,0]
    #yp = data[:,1]
    #prod = data[:,2]
    #turb_trans = data[:,3]
    #visc_trans = data[:,4]
    #press_strain = data[:,5]
    #press_trans = data[:,6]
    eps = data[:,7]
    #balance = data[:,8]

    return eps
    
