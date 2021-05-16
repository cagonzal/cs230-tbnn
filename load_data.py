import numpy as np 

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
