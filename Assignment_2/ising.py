import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   #for my plots to work in terminal commands
import matplotlib.pyplot as plt
from ising_loops_cpp import loop_c #use cython so it compiles faster


'''
for cython to work do 
python -m pip install numpy matplotlib cython setuptools
python setup.py build_ext --inplace 
'''

'''
HOW TO RUN:

You can do : python ising.py
This will run all three questions using the full simulation settings.

To run only one question at a time:
    python ising.py --question 1
    python ising.py --question 2
    python ising.py --question 3

To do a short test run:
    python ising.py --quick

To do a short test run for only one question:
    python ising.py --question 1 --quick
    python ising.py --question 2 --quick
    python ising.py --question 3 --quick

The lattice size can also be changed using --L. For example:
    python ising.py --L 40

or:
    python ising.py --question 2 --L 40
If no lattice size is given, L = 30 is used by default.

The figures and CSV data files are saved in the "outputs" folder

Note: The print statements after plt.show() were added after the code ran and graphs were made. 
Those are my conclusions and discussion on what the results show
'''

saveto = Path('outputs')
saveto.mkdir(exist_ok=True)
SEED = 161616 #random number used to initialise the random number generator
#I learned the seed helps recreate results, since the random number generator will give back the same result everytime for a given seed
#This might be helpful when correcting my results to compare to attached graphs 

#Setup all my functions, easier to call later

def initial_spins(L, rng, aligned=None): #with aligned you can specify a direction for the spin, needed for question 3
    #Creates my lattice, with spin +1 or -1
    if aligned is not None:
        return np.full((L, L), aligned) #this uses the aligned argument to start it a certain way
    return rng.choice([-1, 1], size=(L, L)) #randomize initial spin

def step(spins, T, H, rng):
    #Do one monte carlo sweep of ising lattice by calling the cython function
    L = len(spins) #size 2x2, 3x3 etc... so then total number of spins is L^2
    # One random number for each spin
    rando = rng.random((L, L)) #2D arrays of numbers between 0-1
    #this is so we decide wether to flip the spin, its what we compare the later calculated P to
    loop_c(spins, L, T, rando, H) #call cython function

def magnetisation(spins):
    #I tried to code the mag at first but realise it is simply the mean calculation, so this will save some time in compiling. 
    #could get rid of this function but helps me understand the flow better later on...
    return spins.mean()

def energy_per_spin(spins, H=0.0): #the H argument is for question 3, we need to be able to set mag contribution
    right = np.roll(spins, -1, axis=1) #np.roll addresses the edges effect, it wraps values around when they reach the edge
    down = np.roll(spins, -1, axis=0) #axis 1 is along columns and 0 is for rows
    #also, we only need right and down because if every particle does that then we have accounted for all directions
    E = -np.sum(spins * (right + down)) #calculate the interaction energy, counting each pair once
    E -= H * np.sum(spins)  #include the energy from the external magnetic field
    return E / spins.size #and then divide by number of spins

'''
QUESTION 1

Following the procedure in the notes, write a program which computes the magnetisation
per spin of a square grid of N Ising spins for a given temperature, where each spin is
randomly assigned a spin in the first instance. Using this program, create a plot of the
magnetisation versus Monte Carlo time for: T=1, T=2, T=2.22, T=2.29, T=3, T=5. You
should run the code for a minimum of 1000 Monte Carlo times.(5 marks)
'''
def question1(L, n_steps, seed):
    rng = np.random.default_rng(seed)
    temperatures = [1.0, 2.0, 2.22, 2.29, 3.0, 5.0] #given in question
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True) #plot them all together
    for ax, T in zip(axes.ravel(), temperatures): #ravel gives me a 1D array for my plots
        spins = initial_spins(L, rng) #make my lattice
        history = np.empty(n_steps + 1) #make an empty array the size we need 
        history[0] = magnetisation(spins) #set first number to be intial array of magnetisations
        for t in range(1, n_steps + 1): #Monte-Carlo simulation
            step(spins, T, 0, rng) # no H for this question
            history[t] = magnetisation(spins) #measure new magnetisation and store
        ax.plot(np.arange(n_steps + 1), history, linewidth=0.8) #monte carlo time vs magnetisation
        ax.set_title(f'T = {T}')
        ax.set_ylim(-1.05, 1.05) #magnetisation lies between 1 to -1 so i go a little beyond it
        ax.grid(alpha=0.25)
        np.savetxt(saveto/ f'q1_T{T}.csv', np.column_stack([np.arange(n_steps+1), history]),
                   delimiter=',', header='Monte_Carlo_time,magnetisation_per_spin') #save numerical result, i was doing this in case it took long and i wanted to change the way it plotted. So i had the results and didnt have to run again
    fig.supxlabel('Monte Carlo sweeps')
    fig.supylabel('Magnetisation per spin, M')
    fig.suptitle('Q1: Ising magnetisation versus Monte Carlo sweeps')
    fig.tight_layout()
    fig.savefig(saveto/ 'q1_magnetisation_time.png', dpi=180) #save graph
    plt.show()
    print('At low temperatures (T = 1 and T = 2), the spins rapidly settle into a stable ordered '
    'state with magnetisation close to M = -1. Near the critical region, T = 2.22 and T = 2.29 '
    'show much larger fluctuations and transitions between the two ordered states. T = 2.22 '
    'moves from a mainly negative state to a positive one, while T = 2.29 shows the opposite '
    'behaviour, spending much of the run positive before later becoming negative. At higher '
    'temperatures (T = 3 and T = 5), the magnetisation fluctuates around M = 0, indicating a '
    'disordered state with no long-term alignment.'
)
    print('Q1 done!')

'''
QUESTION 2  

By time-averaging the magnetisation, the total energy per spin of the system, and the
heat capacity per spin for a given temperature, create a plot of the magnetisation, total
energy, and heat capacity of the system as a function of temperature. I suggest using the
temperature range of T=range(1,5,0.03). The heat capacity is given by
C =(ΔE)^2/kT^2
where the variance in the energy is givem by (ΔE)^2= ⟨E^2⟩ - ⟨E⟩^2
and where ⟨⟩ represents the time average of a value. (5 marks)
'''
#Basically-> pick a temp, let lattice settle, measure M and E repeatedly, average measurements, calc heat capacity, repeat for next temp
def question2(L, equil_steps, sample_steps, seed, temp_step):
    rng = np.random.default_rng(seed)
    temperatures = np.arange(1.0, 5.0, temp_step) 
    results = []
    for idx, T in enumerate(temperatures):
        spins = initial_spins(L, rng)
        for i in range(equil_steps): #let lattice evolve, Monte carlo sim, equil_steps sets how long it does this
            step(spins, T, 0, rng)
        ms = np.empty(sample_steps) #empty array that will take mag measurements 
        es = np.empty(sample_steps)  #same for energy per spin
        for t in range(sample_steps):
            step(spins, T, 0.0, rng) #Now do the loop again, but we actually meausre the system after every sweep
            ms[t] = magnetisation(spins)
            es[t] = energy_per_spin(spins)
        #Time averaging
        mean_abs_m = np.mean(np.abs(ms))  #use abs M, because system might settle at +1 or -1 and then the average gives 0 even though the system is very magnetised
        mean_signed_m = np.mean(ms) #i also store it without doing since this might've been what the question actually wanted
        mean_e = np.mean(es)
        #we did es[t] =energy_per_spin(spins) so es us per spin not total
        #so its e = E/N
        #so then C/N = N(heat capacity eq), and spins.size = N
        heat_capacity = spins.size * (np.mean(es**2) - mean_e**2) / (T**2) #let k=1 since we are working in dimensionless units for all variables (T,H)
        results.append([T, mean_abs_m, mean_signed_m, mean_e, heat_capacity]) #if heat capacity is negative return 0
        if idx % 20 == 0: #shows progress, but every 20 temps so it doesnt continuoysly print
            print(f'Q2: {idx+1}/{len(temperatures)} temperatures')
    results = np.asarray(results)
    np.savetxt(saveto/ 'q2_temperature_data.csv', results, delimiter=',', #again save everything
               header='T,mean_absolute_M,mean_signed_M,energy_per_spin,heat_capacity_per_spin')
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True) #and plot
    labels = ['Mean |M| per spin', 'Mean energy per spin', 'Heat capacity per spin']
    for col, ax, label in zip([1, 3, 4], axes, labels): #i didnt ravel because its not a 2d array of graphs so no need
        ax.plot(results[:, 0], results[:, col], '.-', markersize=3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('Temperature, T')
    fig.suptitle('Q2: Temperature dependence of Ising model')
    fig.tight_layout()
    fig.savefig(saveto / 'q2_temp_dependance.png', dpi=180)
    plt.show()
    print('At low temperature the mean absolute magnetisation is close to 1 and the energy per '
    'spin is close to -2, showing that the spins are strongly ordered. Around T = 2.3 the '
    'magnetisation decreases and the energy increases rapidly, while the heat capacity '
    'reaches a clear maximum. These three features identify the critical region of the Ising '
    'model. Above this temperature the magnetisation approaches zero and the system becomes '
    'increasingly disordered.')
    print('Q2 done!')

'''
QUESTION 3

Imagine the system is exposed to an external magnetic field H which either points up or
down. This changes the energy of a given spin to
E_i = -2J∑s_is_j - Hs_i
where H has been re-normalised. 
As shown in the lectures, this leads to the
identification of hysteresis effects when the temperature of the bath is low as the external
field is either swept from low to high values or high to low. Create a plot of M versus H
for T = 0.25 and T = 5, showing that this leads to hysteresis effects below the critical
temperature but not above it. 
Note that when sweeping from negative H to positive H
you'll need to start with a spin lattice where all spins are pointed down, and when
sweeping from positive to negative you'll need to start with a lattice where all spins are
pointed up. (5 marks)
'''

#So basically start with all spins down and slowly increase mag field H from negative to positive, then do the same with spin up and decrese H instead. And then compare.

def field_sweep(L, T, fields, steps_per_field, initial, rng):
    #sweeps mag field while keeping the same lattice
    spins = initial_spins(L, rng, aligned=initial) #create my lattice with the initial spin either all up or all down
    mags = []
    for H in fields: #loop through every mag field value, which is the increasing or decreasing arrays
        for i in range(steps_per_field): #same stuff we did before, do n times of a monte carlo sweep for each mag value
            step(spins, T, H, rng) #basically we dont recreate the lattice for every sping, histeresis remembers the previous state
        mags.append(magnetisation(spins))
    return np.asarray(mags)

def question3(L, steps_per_field, seed, field_max, field_step):
    rng = np.random.default_rng(seed)
    up = np.arange(-field_max, field_max, field_step) #create increasing mag field
    down = up[::-1] #reverse the up array for down
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True) 
    for ax, T in zip(axes, [0.25, 5.0]): #loop through my two temp
        M_up = field_sweep(L, T, up, steps_per_field, -1, rng) #first hysterisis step
        M_down = field_sweep(L, T, down, steps_per_field, +1, rng)
        ax.plot(up, M_up, 'o-', ms=2, label='H increasing')
        ax.plot(down, M_down, 'o-', ms=2, label='H decreasing')
        ax.set_title(f'T = {T}')
        ax.set_xlabel('External magnetic field, H')
        ax.grid(alpha=0.25)
        ax.legend()
        np.savetxt(saveto/ f'q3_T{T}_increasing.csv', np.column_stack([up, M_up]), #save my sim values like before
                   delimiter=',', header='H,M', comments='')
        np.savetxt(saveto/ f'q3_T{T}_decreasing.csv', np.column_stack([down, M_down]),
                   delimiter=',', header='H,M', comments='')
    axes[0].set_ylabel('Magnetisation per spin, M')
    fig.suptitle('Q3: Ising magnetic hysteresis')
    fig.tight_layout()
    fig.savefig(saveto/ 'q3_hysteresis.png', dpi=180) #save image
    plt.show()
    print('At T = 0.25 there is a clear hysteresis loop. The magnetisation remains near its '
    'previous value even after the magnetic field changes sign, and only reverses when the '
    'opposing field becomes sufficiently strong. This shows that the low-temperature system '
    'retains a memory of its previous state. At T = 5 the increasing and decreasing field '
    'curves almost completely overlap and the magnetisation changes smoothly through zero, '
    'showing that significant hysteresis is absent above the critical temperature.')
    print('Q3 done!!')

#Note: the quick control was made before implementation with cython, byusing cython the simulation takes a few seconds, even for all questions, allowing me to even go up in number of sweeps, so the quick was rendered not that useful

def main(): #controls how i call it in a terminal
    p = argparse.ArgumentParser() 
    p.add_argument('--question', choices=['1','2','3','all'], default='all') #you can run each question separately
    p.add_argument('--quick', action='store_true', help='Fast debugging run') #basically use smuch smaller simulations to run the whole thing, was used to check if it worked
    p.add_argument('--L', type=int, default=30, help='Square lattice side length') #lets me choose the L size in terminal
    a = p.parse_args()
    if a.L < 3:
        p.error('You need L >= 3!!')
    if a.quick:
        n1, equil, sample, temp_step, dwell, field_step = 50, 50, 80, 0.3, 2, 0.5 #make sim very short if we use the quick option
    else:
        n1, equil, sample, temp_step, dwell, field_step = 5000, 1000, 5000, 0.02, 50, 0.05 #run for 5000 sweeps, question 2 has 1000 sweeps of settle time, use the temp step given of 0.3
    if a.question in ('1','all'):                                                           #question 30 does 50 sweeps at each magnetic field and changes the mag field by steps of 0.05
        question1(a.L, n1, SEED)
    if a.question in ('2','all'):
        question2(a.L, equil, sample, SEED + 1, temp_step) #slightly diff seed for question 2 and 3
    if a.question in ('3','all'):
        question3(a.L, dwell, SEED + 2, 3.0, field_step)
    print('Finished!!')


if __name__ == '__main__':
    main()
