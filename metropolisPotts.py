from __future__ import division
import numpy as np
from numpy.random import rand
import numba
from numba import jit, autojit, vectorize, double

# q states Potts model - select number of states
q = 3
qstates = np.arange(q)+1
allowedstates = np.exp((2*qstates*(np.pi)*1j)/q)

@jit
def kroenecker(x1, x2):
    if (x1 == x2):
        return 1
    else:
        return 0

@jit
def roll(z):
    z_new = z
    while z == z_new:
        z_new = np.exp((2*(np.random.randint(q)+1)*(np.pi)*1j)/q)
    return z_new

    
@jit
def initstate(N):                                                                                       
    # GENERATE INITIAL SPINS
    init1 = np.random.randint(3, size=(N,N))+1
    init = np.exp((2*init1*(np.pi)*1j)/q)                                                         
    # GENERATES ARRAY OF SIZE N BY N OF Q STATES
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
            z_new = roll(z)
                                                                                       
            # GET OLD AND NEW Z VAL
            
            ESumBefore  = -(kroenecker(z,lattice[(x+1)%N,y]) + kroenecker(z,lattice[x,(y+1)%N]) + kroenecker(z,lattice[(x-1)%N,y]) + kroenecker(z,lattice[x,(y-1)%N]))
            ESumAfter   = -(kroenecker(z_new,lattice[(x+1)%N,y]) + kroenecker(z_new,lattice[x,(y+1)%N]) + kroenecker(z_new,lattice[(x-1)%N,y]) + kroenecker(z_new,lattice[x,(y-1)%N]))     
            dE = ESumAfter - ESumBefore                                                                             
            # CALCULATE ENERGY CHANGE (dE)
            if dE < 0:                                
                z = z_new                                                                                 
                # ACCEPT FLIP IF dE < 0
            elif rand() < np.exp(-dE*beta):
                z = z_new                                                                                 
                # ACCEPT FLIP WITH PROBABILITY OF e^(=dE/T)
            lattice[x,y] = z                                                                            
            # ASSIGN NEW LATTICE POINT VALUE
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
            Zkro = kroenecker(Z,lattice[(x+1)%N, y]) + kroenecker(Z,lattice[x,(y+1)%N]) + kroenecker(Z,lattice[(x-1)%N, y]) + kroenecker(Z,lattice[x,(y-1)%N])    
            E1 += -Z1*Z 
            energy = E1/4.                                                                                             
    return energy

@jit
def latticeMagnetisation(lattice):
    magnetisation = np.sum(lattice)
    return magnetisation

temppoints  = 10                                                                                       
# NUMBER OF POINTS IN TEMPERATURE RANGE
N           = 32                                                                                        
# LATTICE LENGTH
equilibrium = 1024                                                                                      
# NUMBER OF METROPOLIS RUNS TO REACH EQUILIBRIUM
montecarlo  = 1024                                                                                      
# NUMBER OF METROPOLIS RUNS TO PERFORM CALCULATIONS
T           = np.linspace(1.50, 3.50, temppoints)
E           = np.zeros(temppoints)
M           = np.zeros(temppoints)
C           = np.zeros(temppoints)
X           = np.zeros(temppoints)

n1          = 1.0/(montecarlo*N*N)
n2          = 1.0/(montecarlo*montecarlo*N*N) 

lattice = initstate(N)
flatlattice = np.ravel(lattice)
LatticeList = [flatlattice]
MagList = [0]
TempList = [0]

numberofconfigs = 10                                                                                  
# NUMBER OF GENERATED ARRAYS PER TEMPERATURE POINT FOR TRAINING

for i in range(numberofconfigs):
    for tpoints in range(temppoints):                                                                   
        # MAIN CODE BLOCK
        E1 = M1 = E2 = M2 = 0
        lattice = initstate(N)
        beta =1.0/T[tpoints]
    
        for x in range(equilibrium):                                                                    
            # EQUILIBRIATE
            metropolis(lattice, beta)                                  

        for y in range(montecarlo):                                                                     
            # CALCULATE
            metropolis(lattice, beta)          
        
            Ene = latticeEnergy(lattice)                                          
            Mag = latticeMagnetisation(lattice)                                                         

            E1 = E1 + Ene
            M1 = M1 + Mag
            M2 = M2 + Mag*Mag 
            E2 = E2 + Ene*Ene
                       
        flatlattice = np.ravel(lattice)                                                                 
        # SAVE LATTICE TO NUMPY ARRAY
        
        LatticeList = np.concatenate((LatticeList, [flatlattice]))          
        Mg1 = abs(n1*M1)
        Mg = np.round(Mg1, 1)                                
        MagList = np.concatenate((MagList, [Mg]))                   
        temp = np.round(T[tpoints],1)
        TempList = np.concatenate((TempList, [temp]))
        print("Recorded lattice configuration #", tpoints, " of ", temppoints, " in cycle #", i, " of ", numberofconfigs)


np.save("configs.npy", LatticeList)
np.save("maglabels.npy", MagList)
np.save("templabels.npy", TempList)