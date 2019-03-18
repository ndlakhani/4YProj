from __future__ import division
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

N = 16                                                              # SIZE OF SYSTEM

Configs = np.loadtext("configs.txt")                                # LOAD LATTICE CONFIGURATIONS
maglabels = np.loadtxt("maglabels.txt")                             # LOAD MAGNETISATION LABELS
templabels = np.loadtxt("templabels.txt")                           # LOAD TEMPERATURE LABELS

print("Loaded 2D Ising Lattice configurations for training")        # TRACKING OUTPUT - DECLARE SUCCESSFUL IMPORT

