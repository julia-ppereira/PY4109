import matplotlib
matplotlib.use("TkAgg") #so the graph shows when i run my file in bash

import numpy as np
import matplotlib.pyplot as plt

'''
QUESTION 1- Part 3

Distributed with this problem set is a time series of a binary star system. Verify whether
the condition to apply a FT to these data are met. If it is not, write a piece of code which
can manipulate the data to meet this condition. (2 marks)


Notes: The data must be UNIFORMLY SAMPLED. The Fourier transform requires the data to be uniformly sampled, meaning that the time interval deltat between consecutive measurements must remain constant.
'''


data = np.loadtxt("FOAqr.dat")

t = data[:, 0]
y = data[:, 1]
dt = np.diff(t) #interval between consecutive measurements
N = len(t)

if np.allclose(dt, dt[0]): #check wether all the time intervals are equal
    print("The data is uniformly sampled! yay!")
    print("Time step =", np.mean(dt))
else:
    print("The data is not uniformly sampled :((")
    #this will handle if data is not uniform 

    t_uniform = np.linspace(t[0], t[-1], len(t)) #create mty evenly spaced t interval
    y_uniform = np.interp(t_uniform, t, y) #interpolare the measuemtements onto new time values (interp was mentioned in class as something we could use :))


'''
WHAT I GOT:

The data is uniformly sampled! yay!
Time step = 0.00231504

'''

######################################################################################################

'''
QUESTION 1- Part 4

The Fourier Transform can be used as a high pass or low pass filter to isolate signals of
interest. The time series discussed in the previous problem displays multiple
periodicities. These can be broken down into two main components: a series of low
frequency components (f<20 per day) due to the orbital motion of the companion star in
the binary, and some high frequency components (f~68 cycles per day). By using
appropriate frequencies filters, produce a plot of each of the main signals versus time.
You can use a FFT package if necessary (np.fft in python or fftw3.h in C++). (5 marks)

'''

# two main groups
# f < 20 from the binaries orbital motion and
# f ~ 68 for high frequency components

# as it is the low and high frequencies are together we want to separate them so get the signal we need
# so y(t) -> Y(t) -> Y(t) FILTERED -> y(t) filtered
dt = np.mean(np.diff(t))  # i need a single number not an array

Y = np.fft.fft(y)  #gives me fourier coefficients
freq = np.fft.fftfreq(N, d=dt)  #frequency corresponding to each coefficient
positive = freq > 0 #only positive because of mirroring

f_pos = freq[positive]
Y_pos = np.abs(Y[positive])

# With this you can see the mirroring, instead of //2 like before we just have to plot the positive results
plt.figure(figsize=(10, 5))
plt.plot(freq, np.abs(Y))
plt.xlabel("Frequency (cycles/day)")
plt.ylabel(r"$|Y(f)|$")
plt.title("Fourier Spectrum of FO Aqr")
plt.tight_layout()
plt.savefig("q1_4_extra_DFT.png",dpi=300,bbox_inches="tight")
plt.show()

#now we plot just the relevant data, ignoring the mirroring. This lets me see where the peaks are so I can choose my filters 
plt.figure(figsize=(10, 5))
plt.plot(freq[positive], np.abs(Y[positive]))
plt.xlabel("Frequency (cycles/day)")
plt.ylabel(r"$|Y(f)|$")
plt.title("Fourier Spectrum of FO Aqr")
plt.tight_layout()
plt.savefig("q1_4_1_DFT.png",dpi=300,bbox_inches="tight")
plt.show() #you can see the peaks at around less than 20 and then again at more than 68

#keep frequencies below 20 cycles/day
Y_low = Y.copy()
Y_low[np.abs(freq) >= 20] = 0
y_low = np.fft.ifft(Y_low).real #back into the time domain

#isolate high-frequency components around 68 cycles/day 
#i took 50-75 from the plot i made, thats where the peaks where
Y_high = Y.copy()

high_mask = ((np.abs(freq) >= 50) &
             (np.abs(freq) <= 75))

Y_high[~high_mask] = 0
y_high = np.fft.ifft(Y_high).real

fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax[0].plot(t, y_low)
ax[0].set_ylabel("Signal")
ax[0].set_title("Low-frequency orbital signal (f < 20 cycles/day)")
ax[1].plot(t, y_high)
ax[1].set_xlabel("Time (days)")
ax[1].set_ylabel("Signal")
ax[1].set_title("High-frequency signal (~68 cycles/day)")

plt.tight_layout()
plt.savefig("q1_frequency_filters.png", dpi=300)
plt.show()

'''
QUESTION 1 - PART 5
 The two main periods in the frequency spectrum of the previous example are at a
frequency of ∼ 5 cycles per day (which we will label as Ω) and ∼ 69 cycles per day
(which we will label as ω). Identify the relationships between all other significant peaks
and these two periods in the spectrum and submit a plot with all of these peaks clearly
labelled (an example is given below with Ω and ω labelled). (2 marks)

'''

from scipy.signal import find_peaks # i will find peaks using this 

peaks, properties = find_peaks(Y_pos, height=10000)
peak_freqs = f_pos[peaks]
peak_heights = Y_pos[peaks]
for f, A in zip(peak_freqs, peak_heights):
    print(f"{f:.2f} cycles/day   amplitude = {A:.0f}")

# with the peaks i found previously i can plot my lines on them, since they showed me omega and Omega (capital and non capital o)
# i expected these at 5 and 69, but the peak finder gave me 4.98 and 68.95, which worked out perfectly for the combinations :)

Omega = 4.98
omega = 68.95

expected_freqs = np.array([
    Omega/3,
    Omega,
    2*Omega,
    3*Omega,
    4*Omega,
    5*Omega,
    6*Omega,
    omega - 2*Omega,
    omega - Omega,
    omega,
    omega + Omega,
    2*omega - Omega
])

expected_labels = np.array([
    r"$\frac{1}{3}\Omega$",
    r"$\Omega$",
    r"$2\Omega$",
    r"$3\Omega$",
    r"$4\Omega$",
    r"$5\Omega$",
    r"$6\Omega$",
    r"$\omega-2\Omega$",
    r"$\omega-\Omega$",
    r"$\omega$",
    r"$\omega+\Omega$",
    r"$2\omega-\Omega$"
])


plt.figure(figsize=(13, 6))

plt.plot(f_pos, Y_pos, label = "Re($Fe_k$)")

plt.xlim(0, 150)

plt.xlabel("Frequency (cycles/day)")
plt.ylabel("FFT Amplitude")
plt.legend()
plt.title("Frequency Spectrum of FO Aqr")


# Go through each frequency and its corresponding label
for f, label in zip(expected_freqs, expected_labels):

    i = np.argmin(np.abs(f_pos - f)) #fint the fft frequency closest to the expected frequency
    A = Y_pos[i] #then i get the amplitude of that frequency
    b = 2000 #this is to shift my label up

    if A < 17000:#this is because the label was on top of the line for some of the lower amplitudes 
        i = i + 1 #i think it happened since the lower amplitudes have wider peaks, so I wasnt getting encessarily the very top of it 
        A = Y_pos[i] #the top of it seems to be at the next index, thats why the +1 in the previous line 
        b = 20000 #b is just by how much i shift it up, needed more for these ones because one of the peaks still misbehaved and plotted on top of my line 

    plt.axvline(f, linestyle="--", alpha=0.4)
    plt.text( #this is the label then
        f,
        A + b,
        label,
        rotation=90,
        ha="center",
        va="bottom")


plt.tight_layout()
plt.savefig("q1_5.png", dpi=300)
plt.show()