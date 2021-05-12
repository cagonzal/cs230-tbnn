import numpy as np 
import matplotlib.pyplot as plt

filepath = '../re550/LM_Channel_0550_mean_prof.dat'

count = 0
data = np.empty([192, 6])
with open(filepath) as fp:
    for line in fp:
        if line[0] != "%":
            proc_line = (line.strip()).split()
            data[count,:] = np.asarray(proc_line,dtype=np.float)
            count += 1

# y/delta, y+, U, dU/dy, W, P


plt.figure()
plt.plot(data[:,0], data[:,2])

plt.savefig('test.jpg')
