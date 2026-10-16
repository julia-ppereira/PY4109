import matplotlib
matplotlib.use("TkAgg") #so the graph shows when i run my file in bash

import numpy as np
import matplotlib.pyplot as plt

'''
QUESTION 1 - Part 1

Write a piece of code which calculates the discrete Fourier transform. Verify your code
works by calculating the Fourier Transform of y(t) = cos(2π2t). You are free to define
your own internals and time resolution for t. The output of this program should be a plot
of y(t) versus t and a plot of Y (f) versus f. The plot of the Fourier spectrum should also
have the known periods marked with vertical lines. (4 marks)

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

dt=0.01
t = np.arange(0, 4, dt)
y = np.cos(2 * np.pi * 2 * t)

f, Y = DFT(t, y)

fig, ax = plt.subplots(1, 2, figsize=(12, 4)) #plotting two graphs in one figure
#y(t) against t
ax[0].plot(t, y)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("y(t)")
ax[0].set_title(r"$y(t)=\cos(2\pi 2t)$")
#fourier spectrum

ax[1].plot(f[:len(f)//2], np.abs(Y[:len(Y)//2])) #// will give me a round number after division, python arrays need to be integers
#also! i am dividing by two here because: when y(t) is real valued, the second half contains mirrored/conjugate information from the first half
ax[1].axvline(2, linestyle="--",color="red", label="Expected: f = 2 Hz (T = 0.5 s)") #mark the known frequency we ecpect to get 
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel(r"$|Y(f)|$")
ax[1].set_title("Positive-Frequency DFT Spectrum")
ax[1].legend()
plt.tight_layout()
plt.savefig("q1_DFT.png",dpi=300,bbox_inches="tight")
#plt.show()

#If I wanted to plot like mark showing imaginary and and complex components:
r'''
ax[1].scatter(f[:len(f)//2],
    np.real(Y[:len(Y)//2]),
    label=r"$\mathrm{Re}(Y_k)$"#,s=3
)

ax[1].scatter(
    f[:len(f)//2],
    np.imag(Y[:len(Y)//2]),
    label=r"$ \mathrm{Im}(Y_k) $", marker = "x"#, s=3
)

ax[1].axvline(
    2,
    linestyle="--", color="red",
    label="Expected frequency = 2 Hz"
)

ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel(r"$Y_k$")
ax[1].set_title("Discrete Fourier Transform")
ax[1].legend()
plt.tight_layout()
plt.show()
'''

'''
QUESTION 1- Part 2

Compute the DFT for y(t) = cos(2π2t) + sin(2π4t) for the following intervals
(min,max,step):

1. t = range(0,4,0.5)
2. t = range(0,4,0.1)
3. t = range(0,20,0.1)

Using these results, comment on the accuracy of the recovered periods as both a
function of the number of samples and the time step between the samples. (5 marks)

'''

#signal containing frequencies of 2 Hz and 4 Hz, these correspond to periods of 0.5 s and 0.25 s.

t1 = np.arange(0, 4, 0.5)
t2 = np.arange(0, 4, 0.1)
t3 = np.arange(0, 20, 0.1)

y1 = np.cos(2*np.pi*2*t1) + np.sin(2*np.pi*4*t1)
y2 = np.cos(2*np.pi*2*t2) + np.sin(2*np.pi*4*t2)
y3 = np.cos(2*np.pi*2*t3) + np.sin(2*np.pi*4*t3)

f1, Y1 = DFT(t1, y1)
f2, Y2 = DFT(t2, y2)
f3, Y3 = DFT(t3, y3)


#in one figure for better comparison
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax = ax.ravel()


fig.suptitle(r"DFT of $y(t)=\cos(2\pi 2t)+\sin(2\pi 4t)$",fontsize=16)

#dt = 0.5, 4

ax[3].plot(f1[:len(f1)//2],np.abs(Y1[:len(Y1)//2]))
ax[3].axvline(2, linestyle="--", color="red", label="2 Hz (T = 0.5s)")
ax[3].axvline(4, linestyle="--", color="green", label="4 Hz (T = 0.25s)") #two known freq
ax[3].set_xlabel("Frequency (Hz)")
ax[3].set_ylabel(r"$|Y(f)|$")
ax[3].legend()
ax[0].plot(t1, y1,"o-")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("y(t)")


#dt = 0.1, 4

ax[4].plot(f2[:len(f2)//2],np.abs(Y2[:len(Y2)//2]))
ax[4].axvline(2, linestyle="--", color="red",label="2 Hz (T = 0.5s)") #two known freq
ax[4].axvline(4, linestyle="--", color="green",label="4 Hz (T = 0.25s)")
ax[4].set_xlabel("Frequency (Hz)")
ax[4].set_ylabel(r"$|Y(f)|$")
ax[4].legend()
ax[1].plot(t2, y2,"o-")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("y(t)")


#dt = 0.1, 20 

ax[5].plot(f3[:len(f3)//2],np.abs(Y3[:len(Y3)//2]))
ax[5].axvline(2, linestyle="--", color="red",label="2 Hz (T = 0.5s)") #two known freq
ax[5].axvline(4, linestyle="--", color="green",label="4 Hz (T = 0.25s)")
ax[5].set_xlabel("Frequency (Hz)")
ax[5].set_ylabel(r"$|Y(f)|$")
ax[5].legend()
ax[2].plot(t3, y3,"o-")
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("y(t)")


ax[3].legend(loc="upper left")
ax[4].legend(loc="upper left")
ax[5].legend(loc="upper left")

ax[0].set_title(r"$\Delta t=0.5$ s, $N=8$") #titles describing each sampling case
ax[1].set_title(r"$\Delta t=0.1$ s, $N=40$")
ax[2].set_title(r"$\Delta t=0.1$ s, $N=200$")
# ax[3].set_title(r"$y(t)$") #then bottom plots 
# ax[4].set_title(r"$y(t)$")
# ax[5].set_title(r"$y(t)$")

plt.tight_layout()
plt.savefig("q1_2_DFT.png",dpi=300,bbox_inches="tight")
plt.show()

'''
Comments on results:

Signal has expected frequencies 2Hz and 4Hz.

The t = np.arange(0,4,0.5) means arranging it so that we take a measurement every 0.5 seconds (Deltat=0.5). 
The sampling frequency is how many measurements we take per second, and thats 1/Deltat = 1/0.5 = 2Hz, so two measurements per second.
In this range we are taking 8 measurements N=4/0.5=8. The DFT then produces 8 Complex fourier coeffiecients, which can contai 2N independent real numbers. 
This comes from the fact that our original y(t) is real, so the DFT has conjugate symmmetry.
Second half mirrors the first half. It is why we do division here ax[1].plot(f[:len(f)//2], np.abs(Y[:len(Y)//2]))
We define frequencies as nu = m/(N*dt)
So the largest independency frequency index is approx m=N/2 because of the mirroring
so then our f_max =  (N/2)/(N*dt) = 1/2*dt which is the Nyquist frequency!

If we apply to the first scenario:

f_max =  (8/2)/(8*0.5) = 1/2*0.5 = 1 Hz
Which is what we see in the first DFT plot, the data stops around 1Hz 
In fact, the signal itself is basically a flat line. 
That's because we happened to sample this particular signal at really unfortunate points. 
The real signal is oscillating between your measurements, but we are missing those oscillations.
This points to undersampling. 

If we reduce to 0.1, like in the second plot

f_max =  (40/2)/(40*0.1) = 1/2*0.1 = 5 Hz

and this allows for both expected frequencies of 2Hz and 4Hz

Now when we move to 200 samples, sampling from 0-20s instead of 0-4s, we have the exact same nyquist frequency of 5 (same dt), so this isnt about detecting higher frequencies but observing the signal for longer to get more accurate results.
We are testing frequencies much mroe closely spaced, we test them every deltaf = 1/N*dt which is approx 1/Tobs 
Comparing run 2 and 3

deltaf= 1/40*0.1 =0.25
deltaf = 1/200*0.1 = 0.05

So if the expected frequency were something awkward like 2.13 Hz, the second run of N=40 run could only put a peak on nearby bins such as 2.00 or 2.25 Hz, whereas the longer run has much more finely spaced frequency bins.
This means we have better frequency resolution. 

You can see that in the graph, we have narrower and taller peaks (taller because no normalization by N). The difference in frequency resolution isnt seem as much for this because 2 and 4 happen to fall within the bins for run 2 asd well.

'''
