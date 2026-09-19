import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

'''
QUESTION 1 

Write a piece of code which calculates the discrete Fourier transform. Verify your code
works by calculating the Fourier Transform of y(t) = cos(2π2t). You are free to define
your own internals and time resolution for t. The output of this program should be a plot
of y(t) versus t and a plot of Y (f) versus f. The plot of the Fourier spectrum should also
have the known periods marked with vertical lines. (4 marks)

'''

def DFT(t, y):
    N = len(t) #no of data points, DFT will produce N Fourier coefficients 
    m = np.arange(N) #which time sample we are looking at, this looks like m = [0, 1, 2, ..., N-1]
    nu = m/max(t) #assuming max(t) is approx N \Delta t (done in code used in lecture)
    Y = np.zeros(N, dtype=complex) # preparing array to store Ys

    for n in range(N): #Go through every frequency n that the DFT can test.
        Y[n] = np.sum(y * np.exp(2*np.pi*1j*m*n/N)) #python uses j as complex notation, 1j = 1*i = i

    return nu, Y

dt = 0.01
t = np.arange(0, 4, dt)
y = np.sin(2 * np.pi * 2 * t)

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
ax[1].axvline(2, linestyle="--",color="red", label="Expected frequency = 2 Hz") #mark the known frequency we ecpect to get 
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel(r"$|Y(f)|$")
ax[1].set_title("Discrete Fourier Transform")
ax[1].legend()
plt.tight_layout()
plt.savefig("q1_plot.png", dpi=150)

#If I wanted to plot like mark showinf imaginary and and complex components:
'''
ax[1].scatter(f[:len(f)//2],
    np.real(Y[:len(Y)//2]),
    label=r"$\mathrm{Re}(Y_k)$"#,s=3
)

ax[1].scatter(
    f[:len(f)//2],
    np.imag(Y[:len(Y)//2]),
    label=r"$\mathrm{Im}(Y_k)$", marker = "x"#, s=3
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