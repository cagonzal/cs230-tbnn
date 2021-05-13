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
