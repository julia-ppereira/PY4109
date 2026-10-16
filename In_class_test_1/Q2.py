import matplotlib
matplotlib.use("TkAgg") #so the graph shows when i run my file in bash
from scipy.signal import find_peaks # i will find peaks using this 
import numpy as np
import matplotlib.pyplot as plt

# NOTE: APOLOGIES BUT THIS CODE DOES NOT RUN :(


data = np.loadtxt("Q2.dat")

t = data[:, 0]
y = data[:, 1]
y_error = data[:,2]
dt = np.diff(t) #interval between consecutive measurements
N = len(t)

intervals = np.arange(0, t[-1], 101) #idea here is i would separate my data into 15 evenly spaced intervals in the space I have
#initially i tried to divide by arranging the data, but this did not work as expected 

t_uniform = np.linspace(t[0], t[-1], 15) 
y_uniform = np.interp(t_uniform, t, y) #i now realise interpolate would not work but I will leave it here just in case


def DFT(t, y):
    N = len(t) #no of data points, DFT will produce N Fourier coefficients 
    m = np.arange(N) #which time sample we are looking at, this looks like m = [0, 1, 2, ..., N-1]
    dt = t[1] - t[0] #time interval between consecutive samples assuming data is uniformly samples
    nu = m / (N * dt) #total duration is N*dt, so frequency spacinf has to be 1/(N*dt)
    Y = np.zeros(N, dtype=complex) # preparing array to store Ys

    for n in range(N): #Go through every frequency n that the DFT can test.
        Y[n] = np.sum(y * np.exp(2*np.pi*1j*m*n/N)) #python uses j as complex notation, 1j = 1*i = i

    return nu, Y

dt=dt[0]

f, Y = DFT(t, y)

for t_i, y_i in zip(t_uniform, y_uniform):
    mask = (t >= t_i) & (t <= t_i+1)            #the idea here would be that i mask for t is more than first and less than second, and it would go up from there until it goes through all the points 
    print("======INTERVAL 1=======")
    print(f"Start of interval={t_i}") #this would ideally print my beginning and end of interval 
    print(f"End of Interval={t_i+1}")
    t_use= t[mask] #then i would mask my f and y for the selected time, sinnce theyd be the same size 
    y_use= y[mask]
    f, Y =  DFT(t_use, y_use)
    P = (np.real(Y[:len(Y)//2]))**2 + (np.imag(Y[:len(Y)//2]))**2 #fidnd my power 
    peaks, properties = find_peaks(P, height=3e4)
    peak_freqs = f[peaks]
    peak_heights = P[peaks]
    for peak, X in zip(peak_freqs, peak_heights):
        print(f"{peak:.2f} Hz  Power = {X}, Period = {1/peak:.2f} Hz-1") #if function prints no peaks it does not have any relevant ones at that interval 
