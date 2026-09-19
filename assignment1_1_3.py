import matplotlib
matplotlib.use("TkAgg") #so the graph shows when i run my file in bash

import numpy as np
import matplotlib.pyplot as plt

'''
QUESTION 1- Part 3

Distributed with this problem set is a time series of a binary star system. Verify whether
the condition to apply a FT to these data are met. If it is not, write a piece of code which
can manipulate the data to meet this condition. (2 marks)


Notes: The data must be UNIFORMLY SAMPLES 
'''


data = np.loadtxt("FOAqr.dat")

t = data[:, 0]
y = data[:, 1]

# Calculate the time interval between consecutive measurements
dt = np.diff(t) #interval between consecutive measurements

if np.allclose(dt, dt[0]): #check wether all the time intervals are equal
    print("The data is uniformly sampled! yay!")
    print("Time step =", np.mean(dt))
else:
    print("The data is not uniformly sampled :((")

'''

WHAT I GOT:

The data are uniformly sampled.
Time step = 0.00231504

MEANING:

The data is uniformly sampled yay!!
'''

def DFT(t, y):
    N = len(t) #no of data points, DFT will produce N Fourier coefficients 
    m = np.arange(N) #which time sample we are looking at, this looks like m = [0, 1, 2, ..., N-1]
    dt = t[1] - t[0] #time interval between consecutive samples assuming data is uniformly samples
    nu = m / (N * dt) #total duration is N*dt, so frequency spacinf has to be 1/(N*dt)
    Y = np.zeros(N, dtype=complex) # preparing array to store Ys

    for n in range(N): #Go through every frequency n that the DFT can test.
        Y[n] = np.sum(y * np.exp(2*np.pi*1j*m*n/N)) #python uses j as complex notation, 1j = 1*i = i

    return nu, Y