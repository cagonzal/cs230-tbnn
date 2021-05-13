import numpy as np 

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
