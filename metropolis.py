from __future__ import division
import numpy as np
from numpy.random import rand

def initstate(N):                                                                                       # GENERATE INITIAL SPINS
    init = 2*np.random.randint(2, size=(N,N))-1                                                         # GENERATES ARRAY OF SIZE N BY N OF 1s and -1s
    return init



