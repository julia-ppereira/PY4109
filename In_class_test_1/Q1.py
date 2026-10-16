import matplotlib
matplotlib.use("TkAgg") #so the graph shows when i run my file in bash

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Q1.dat")

t = data[:, 0]
y = data[:, 1]
y_error = data[:,2]
dt = np.diff(t) #interval between consecutive measurements
N = len(t)

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

f, Y = DFT(t, y) #use my function

fig, ax = plt.subplots(1, 2, figsize=(12, 4)) #plotting two graphs in one figure
#y(t) against t
ax[0].plot(t, y)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("y(t)")
ax[0].set_title(r"Raw Data$")

#fourier spectrum

ax[1].scatter(f[:len(f)//2],
    np.real(Y[:len(Y)//2]),
    label=r"$\mathrm{Re}(Y_k)$"#,s=3
)

ax[1].scatter(
    f[:len(f)//2],
    np.imag(Y[:len(Y)//2]),
    label=r"$ \mathrm{Im}(Y_k) $", marker = "x"#, s=3
)


ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel(r"$Y_k$")
ax[1].set_title("DFT- Real and Imaginary Parts")
ax[1].legend()
plt.tight_layout()
plt.savefig("q1_1",dpi=300,bbox_inches="tight")
plt.show()


#PRINT OUT HIGHEST FREQUENCY: 

print("Max frequency:", max(f[:len(f)//2])) 


#Plot of Power spectrum 

P = (np.real(Y[:len(Y)//2]))**2 + (np.imag(Y[:len(Y)//2]))**2

plt.figure(figsize=(10, 5))
plt.plot(f[:len(f)//2], P)
print(P)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Power Spectrum of TIme Series")
plt.savefig("q1_2.png",dpi=300,bbox_inches="tight")
plt.tight_layout()
plt.show()

#Locate 4 highest peaks 

from scipy.signal import find_peaks # i will find peaks using this 

peaks, properties = find_peaks(P, height=3.2e4)
peak_freqs = f[peaks]
peak_heights = P[peaks]
for peak, X in zip(peak_freqs, peak_heights):
    print(f"{peak:.2f} Hz  Power = {X}, Period = {1/peak:.2f} Hz-1")


# NOW PART 5 

P_filter = P.copy()
f_filter = f.copy()

mask1 = ((f_filter[:len(f_filter)//2] >= 0.1) &
             (f_filter[:len(f_filter)//2] <= 0.5))

mask2 = ((f_filter[:len(f_filter)//2]  >= 1.8) &
             (f_filter[:len(f_filter)//2]  <= 2.21))

mask3 = ((f_filter[:len(f_filter)//2]  >= 4.82) &
             (f_filter[:len(f_filter)//2]  <= 5.35))

mask4 = ((f_filter[:len(f_filter)//2]  >= 5.76) &
             (f_filter[:len(f_filter)//2]  <= 6.2))

P_filter[mask1] = 0
P_filter[mask2] = 0
P_filter[mask3] = 0
P_filter[mask4] = 0

print(f_filter)



plt.figure(figsize=(10, 5))
plt.plot(f_filter[:len(f_filter)//2], P_filter)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Power Spectrum of TIme Series")
plt.tight_layout()
plt.savefig("q1_3.png",dpi=300,bbox_inches="tight")
plt.show()

Y_filter = Y.copy() #error because array has to be halved to account for symmetry, ran out of time to fix this 

Y_filter[mask1] = 0
Y_filter[mask2] = 0
Y_filter[mask3] = 0
Y_filter[mask4] = 0

#Take inverse fourier 
y_back = np.fft.ifft(Y_filter) #back into the time domain, i dont have a function for this sorry

y_back_shuf = np.random.shuffle(y_back)

#DO FFT AGAIN
f_shuf, Y_shuf = DFT(t, y_shuf)

#COMPARE TO FFT OF REAL DATA 

fig, ax = plt.subplots(1, 2, figsize=(12, 4)) #plotting two graphs in one figure
#y(t) against t
ax[0].plot(f[:len(f_shuf)//2], Y[:len(Y_shuf)//2])
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel(r"$Y_k$")
ax[0].set_title(r"Shuff FFT$")

#fourier spectrum

ax[1].plot(f[:len(f)//2], Y[:len(Y)//2])
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("y(t)")
ax[1].set_title(r"Raw Data$")


ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel(r"$Y_k$")
ax[1].set_title("Initial FFT")
ax[1].legend()
plt.tight_layout()
plt.savefig("q1_4_1.png",dpi=300,bbox_inches="tight")
plt.show()





#I RAN OUT OF TIME BUT I WOULD:

'''
1. Calculate power series again and compare to the original power series aqcquired 
2. Redo this proces n amount of times with a for loop, Then identify peaks for each run, and give out a number of how often we get the peak at the power series 
3. This percent would then inform me of the level of confidence of this peak 

'''