from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D


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
y_train = label/3.5

test_x = test_dataset
y_test = tlabels/3.5

# PREPARING DATA: RESHAPE LATTICE TO N*N*1, COVERT LABELS TO CATEGORICAL

latticeshape = (N, N, 1)
x_train = x.reshape(x.shape[0], N, N, 1)
x_test = test_x.reshape(test_x.shape[0],N,N,1)

# CREATE MODEL
model = Sequential()

# CONVOLUTION LAYERS
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=latticeshape))

# DROPOUT AND FLATTEN

model.add(Flatten())

# DENSE LAYERS
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

# OUTPUT LAYER
model.add(Dense(1, activation='linear'))

# DISPLAY NETWORK ARCHITECTURE
model.summary()

# COMPILE MODEL
model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['mean_absolute_error', 'mean_squared_error'])

# TRAIN MODEL
history = model.fit(x_train, y_train, batch_size=100, epochs=25, verbose=1, validation_data=(x_test, y_test))

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)

print('Test error:', score[0])

xpredict = np.array(np.load("predictdata.npy"))
xpred = xpredict.reshape(xpredict.shape[0], N, N, 1)

ypredict = model.predict(xpred)

truelabels = np.array(np.load("predictlabels.npy"))
ylabels = truelabels/3.5

order = np.abs(np.sum([xpredict], axis=2))
order = order.reshape(order.shape[1],)
order = order/1024
plt.plot(ypredict,order,'x')
