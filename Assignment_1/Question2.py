import matplotlib
matplotlib.use("TkAgg") #so the graph shows when i run my file in bash

import numpy as np
import matplotlib.pyplot as plt

'''
QUESTION 2- Part 2

Write a Monte Carlo code which simulates isotropic scattering of photons through a uniform slab of thickness:

zmax = 1 
τmax = 10

The photons should all be initialised at the origin with an initial angle of direction aligned with the z axis. They should then move
a distance which is related to a randomly drawn optical depth, and then scatter into a random direction. 

The ending criterion for considering a photon should be when it leaves the atmosphere at z = zmax or back scatters out of the atmosphere. 
The final output of this program should be a histogram of the angles over which photons are emitted. 
Use this code to reproduce Chandresekhar's result from 1960 for a plane atmosphere as shown below (and attached to this assignment as a dat file with columns of angle and
intensity). 

Simulating roughly 100,000 photons should be enough to reproduce the result (depending on your bin size). Note that you'll need to convert from number of photons 
(flux) in a given bin to intensity. This can be done using
Iv/F = NiNμ/ 2N0μi
where Ni is the number of photons in bin i, Nμ is the total number of μ bins you split the data in to, N0 is the total number of photons, and μi is the μ at the centre of the ith bin.
'''

class Photon:

    def __init__(self, zmax=1, taumax=10):

        #start at origin
        self.x = 0
        self.y = 0
        self.z = 0

        #initial direction along z axis
        self.theta = 0
        self.phi = 0

        self.zmax = zmax
        self.taumax = taumax

        self.mu = None #my final outgoing mu, filled when photon escapes

    def scatter(self):

        rnd1 = np.random.uniform(0, 1) #equations from part 1 
        rnd2 = np.random.uniform(0, 1) #we need two xi for each phi and theta (they will scatter randomly in their own way)
        mu = 2*rnd1 - 1
        self.theta = np.arccos(mu)
        self.phi = 2*np.pi*rnd2


    def evolve_photon(self):

        while True:

            rnd = np.random.uniform(0, 1) #get a random number between 0-1
            tau = -np.log(1 - rnd) #sample optical path
            L = tau / self.taumax #convert to real distance

            #move photon
            self.x = self.x + L*np.sin(self.theta)*np.cos(self.phi)
            self.y = self.y + L*np.sin(self.theta)*np.sin(self.phi)
            self.z = self.z + L*np.cos(self.theta)

            #photon escapes through top
            if self.z >= self.zmax:
                self.mu = np.cos(self.theta)
                break
            
            if self.z < 0:
                self.mu=None
                break

            self.scatter()

N = 100000 # even though 10000 did replicate the shape already, 100000 only took about 10 seconds and gave the best graph, so i went with this

data = np.loadtxt(r"Chandrasekhar1960.dat")
angle_chandra = data[:, 0]
intensity_chandra = data[:, 1]

final_mu = []

for i in range(N):

    photon = Photon() #create photon
    photon.evolve_photon() #evolve photon until break condition

    if photon.mu is not None:
        final_mu.append(photon.mu)

final_mu = np.array(final_mu)

N_mu = 20

#create my histogram
counts, bin_edges = np.histogram(final_mu,bins=N_mu,range=(0, 1))
#bin centre for plotting
mu_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

# number of photons in the distribution that they left the atmosphere 
N0 = len(final_mu)

intensity = (counts * N_mu) / (2 * N0 * mu_centres)

# convert the Monte Carlo mu values to angles in degrees
theta_centres = np.degrees(np.arccos(mu_centres))

# sort by increasing angle
order_mc = np.argsort(theta_centres)
theta_centres = theta_centres[order_mc]
intensity = intensity[order_mc]

plt.figure(figsize=(8, 5))

plt.plot(theta_centres,intensity,"o-", label="Monte Carlo")
plt.plot(angle_chandra,intensity_chandra,label="Chandrasekhar (1960)")
plt.xlabel(r"$\theta$ (degrees)")
plt.ylabel(r"$I_\nu/F$")
plt.title("Monte Carlo Comparison with Chandrasekhar (1960)")
plt.xlim(0, 90)
plt.legend()
plt.grid()
plt.savefig("q2_montecarlo.png",dpi=300,bbox_inches="tight")
plt.show()

print("Total simulated:", N)
print("Escaped through top:", N0)
print("Fraction escaping:", N0/N)


'''
Determine appropriate errors for each of your bins, and update your plot accordingly.
Justify your choice of errors. (4 marks)
'''

#poisson error on the number of photons in each bin 
counts_error = np.sqrt(counts)
intensity_error = (counts_error * N_mu) / (2 * N0 * mu_centres) #propagate to intensity
intensity_error = intensity_error[order_mc] #sort them like the rest

plt.figure(figsize=(8, 5))

plt.errorbar(theta_centres,intensity,yerr=intensity_error,fmt="o-",capsize=3,label="Monte Carlo")
plt.plot(angle_chandra,intensity_chandra,label="Chandrasekhar (1960)")
plt.xlabel(r"$\theta$ (degrees)")
plt.ylabel(r"$I_\nu/F$")
plt.title("Monte Carlo Comparison with Chandrasekhar (1960)")
plt.xlim(0, 90)
plt.legend()
plt.grid()
plt.savefig("q2_montecarlo_error.png",dpi=300,bbox_inches="tight")
plt.show()