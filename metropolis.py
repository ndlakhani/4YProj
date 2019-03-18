from __future__ import division
import numpy as np
from numpy.random import rand

def initstate(N):                                                                                       # GENERATE INITIAL SPINS
    init = 2*np.random.randint(2, size=(N,N))-1                                                         # GENERATES ARRAY OF SIZE N BY N OF 1s and -1s
    return init


def metropolis(lattice, beta):                                                                          # MONTE CARLO VIA METROPOLIS
    for a in range(N):
        for b in range(N):
            x = np.random.randint(0,N)                                                                  # SELECT LATTICE POINT X-COORDINATE
            y = np.random.randint(0,N)                                                                  # SELECT LATTICE POINT Y-COORDINATE
            z = lattice(x, y)                                                                           # DEFINE LATTICE POINT VALUE
            z1 = lattice[(x+1)%N,y] + lattice[x,(y+1)%N] + lattice[(x-1)%N,y] + lattice[x,(y-1)%N]      
            prob = 2*z*z1                                                                               # CALCULATE ENERGY CHANGE (dE)
            if prob < 0:                                
                z *= -1                                                                                 # ACCEPT FLIP IF dE < 0
            elif rand() < np.exp(-prob*beta):
                z *= -1                                                                                 # ACCEPT FLIP WITH PROBABILITY OF e^(=dE/T)
            lattice[x,y] = z                                                                            # ASSIGN NEW LATTICE POINT VALUE (either 1 or -1)
    return lattice

def latticeEnergy(lattice):                                                                             # CALCULATE ENERGY OF LATTICE CONFIGURATION
    E1 = 0
    for x in range(len(lattice)):                                                                       # CALCULATE OVER ENTIRE LATTICE - ALL LATTICE POINTS
        for y in range(len(lattice)):                                                                   
            Z = lattice[x,y]
            Z1 = lattice[(x+1)%N, y] + lattice[x,(y+1)%N] + lattice[(x-1)%N, y] + lattice[x,(y-1)%N]      
            E1 += -Z1*Z
            E0 = E1/4.                                                                                  
            energy = E0
    return energy

def latticeMagnetisation(lattice):
    magnetisation = np.sum(lattice)
    return magnetisation

