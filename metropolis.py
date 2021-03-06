from __future__ import division
import numpy as np
from numpy.random import rand
import numba
from numba import jit, autojit, vectorize, double

# MAGNETIC FIELD - SET H TO 0 FOR NO MAGNETIC FIELD
h = 0.5

@jit
def initstate(N):                                                                                       
    # GENERATE INITIAL SPINS
    init = 2*np.random.randint(2, size=(N,N))-1                                                         
    # GENERATES ARRAY OF SIZE N BY N OF 1s and -1s
    return init

@jit
def metropolis(lattice, beta):                                                                          
    # MONTE CARLO VIA METROPOLIS
    for a in range(N):
        for b in range(N):
            x = np.random.randint(0,N)                                                                  
            # SELECT LATTICE POINT X-COORDINATE
            y = np.random.randint(0,N)                                                                  
            # SELECT LATTICE POINT Y-COORDINATE
            z = lattice[x, y]                                                                           
            # DEFINE LATTICE POINT VALUE
            z1 = lattice[(x+1)%N,y] + lattice[x,(y+1)%N] + lattice[(x-1)%N,y] + lattice[x,(y-1)%N]
            H = np.sum([lattice])*h      
            prob = 2*((z*z1)+H)                                                                              
            # CALCULATE ENERGY CHANGE (dE)
            if prob < 0:                                
                z *= -1                                                                                 
                # ACCEPT FLIP IF dE < 0
            elif rand() < np.exp(-prob*beta):
                z *= -1                                                                                 
                # ACCEPT FLIP WITH PROBABILITY OF e^(=dE/T)
            lattice[x,y] = z                                                                            
            # ASSIGN NEW LATTICE POINT VALUE (either 1 or -1)
    return lattice

@jit
def latticeEnergy(lattice):                                                                             
    # CALCULATE ENERGY OF LATTICE CONFIGURATION
    E1 = 0
    for x in range(len(lattice)):                                                                      
         # CALCULATE OVER ENTIRE LATTICE - ALL LATTICE POINTS
        for y in range(len(lattice)):                                                                   
            Z = lattice[x,y]
            Z1 = lattice[(x+1)%N, y] + lattice[x,(y+1)%N] + lattice[(x-1)%N, y] + lattice[x,(y-1)%N]
            H1 = np.sum([lattice])*h      
            E1 += -Z1*Z - H1
            E0 = E1/4.                                                                                  
            energy = E0
    return energy

@jit
def latticeMagnetisation(lattice):
    magnetisation = np.sum(lattice)
    return magnetisation

# NUMBER OF POINTS IN TEMPERATURE RANGE
temppoints  = 1000                                                                                       
# LATTICE LENGTH
N           = 32                                                                                        
# NUMBER OF METROPOLIS RUNS FOR EQUILIBRIATION
equilibrium = 1024                                                                                      
# NUMBER OF METROPOLIS RUNS FOR CALCULATIONS
montecarlo  = 1024                                                                                      

T           = np.linspace(1.50, 3.50, temppoints)
n1          = 1.0/(montecarlo*N*N)

lattice = initstate(N)
flatlattice = np.ravel(lattice)
LatticeList = [flatlattice]
MagList = [0]
TempList = [0]

# NUMBER OF GENERATED ARRAYS PER TEMPERATURE POINT FOR TRAINING
numberofconfigs = 1                                                                                  

for i in range(numberofconfigs):
    for tpoints in range(temppoints):                                                                   
        # MAIN CODE BLOCK
        E = M = 0
        lattice = initstate(N)
        beta =1.0/T[tpoints]
    
        for x in range(equilibrium):                                                                    
            # EQUILIBRIATE
            metropolis(lattice, beta)                                  

        for y in range(montecarlo):                                                                     
            # CALCULATE
            metropolis(lattice, beta)          
        
            Energy = latticeEnergy(lattice)                                          
            Mag = latticeMagnetisation(lattice)                                                         

            E = E + Energy
            M = M + Mag          
                       
        flatlattice = np.ravel(lattice)                                                                 
        
        # SAVE LATTICE TO NUMPY ARRAY       
        LatticeList = np.concatenate((LatticeList, [flatlattice])) 
         
        Mg = abs(n*M)                                       
        MagList = np.concatenate((MagList, [Mg]))                   
        temp = T[tpoints]
        TempList = np.concatenate((TempList, [temp]))
        print("Recorded lattice configuration #", tpoints, " of ", temppoints, " in cycle #", i, " of ", numberofconfigs)


np.save("configs.npy", LatticeList)
np.save("maglabels.npy", MagList)
np.save("templabels.npy", TempList)