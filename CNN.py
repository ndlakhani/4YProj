from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

N = 16                                                                      # SIZE OF SYSTEM

Configs         = np.load("latticelist.npy")                                # LOAD LATTICE CONFIGURATIONS
#maglabels       = np.load("maglabels.txt")                                  # LOAD MAGNETISATION LABELS
templabels      = np.load("templist.npy")                                   # LOAD TEMPERATURE LABELS

print("Loaded 2D Ising Lattice configurations for training")                # TRACKING OUTPUT - DECLARE SUCCESSFUL IMPORT

test_x          = np.array(np.loadtxt("testconfigs.txt"))
test_label      = np.array(np.loadtxt("testtemplabels.txt"))

print("Loaded 2D Ising Lattice configurations for testing")                 # TRACKING OUTPUT - DECLARE SUCCESSFUL IMPORT

