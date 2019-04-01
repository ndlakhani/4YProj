from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import regularizers


# SIZE OF SYSTEM
N = 16                                                                      


# LOAD LATTICE CONFIGURATIONS
train_dataset = np.load("latticelist.npy")

# LOAD MAGNETISATION LABELS (IF UNCOMMENTED)                     
# maglabels = np.load("maglabels.txt")
          
# LOAD TEMPERATURE LABELS                                          
label = np.array(np.load("templist.npy"))
# TRACKING OUTPUT - DECLARE SUCCESSFUL IMPORT
print("Loaded 2D Ising Lattice configurations for training")                

test_dataset = np.array(np.load("testconfigs.npy"))
tlabels = np.array(np.load("testtemplabels.npy"))

# TRACKING OUTPUT - DECLARE SUCCESSFUL IMPORT
print("Loaded 2D Ising Lattice configurations for testing")                 


x = train_dataset
y_train = label/3.6

test_x = test_dataset
y_test = tlabels/3.6

# PREPARING DATA: RESHAPE LATTICE TO N*N*1, COVERT LABELS TO CATEGORICAL

latticeshape = (N, N, 1)
x_train = x.reshape(x.shape[0], N, N, 1)
x_test = test_x.reshape(test_x.shape[0],N,N,1)

# CREATE MODEL
model = Sequential()

# CONVOLUTION LAYERS
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=latticeshape))

# DROPOUT AND FLATTEN

model.add(Flatten())

# DENSE LAYERS
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))

# OUTPUT LAYER
model.add(Dense(1, activation='linear'))

# DISPLAY NETWORK ARCHITECTURE
model.summary()

# COMPILE MODEL
model.compile(loss='mean_absolute_error', optimizer="sgd", metrics=['mean_absolute_error'])

# TRAIN MODEL
history = model.fit(x_train, y_train, batch_size=100, epochs=25, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test error:', score[0])

xpredict = np.array(np.load("predictdata.npy"))
xpred = xpredict.reshape(xpredict.shape[0], N, N, 1)

ypredict = model.predict(xpred)

truelabels = np.array(np.load("predictlabels.npy"))
ylabels = truelabels/3.6
