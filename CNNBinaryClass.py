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
N = 32                                                                      

# LOAD LATTICE CONFIGURATIONS
train_dataset = np.load("latticelist.npy")

# LOAD LATTICE ORDER PARAMETERS                                      
label = np.sum(train_dataset, axis = 1)
label = np.abs(label/(N*N))
label[label>=0.5] = 1
label[label<0.5] = 0

# TRACKING OUTPUT - DECLARE SUCCESSFUL IMPORT
print("Loaded 2D Ising Lattice configurations for training")                

test_dataset = np.array(np.load("testconfigs.npy"))
tlabels = np.sum(test_dataset, axis = 1)
tlabels = np.abs(tlabels/(N*N))
tlabels[tlabels>=0.5] = 1
tlabels[tlabels<0.5] = 0

# TRACKING OUTPUT - DECLARE SUCCESSFUL IMPORT
print("Loaded 2D Ising Lattice configurations for testing")                 


x = train_dataset

test_x = test_dataset

# PREPARING DATA: RESHAPE LATTICE TO N*N*1, COVERT LABELS TO CATEGORICAL

latticeshape = (N, N, 1)
x_train = x.reshape(x.shape[0], N, N, 1)
x_test = test_x.reshape(test_x.shape[0],N,N,1)
y_train = keras.utils.to_categorical(label)
y_test = keras.utils.to_categorical(tlabels)

# CREATE MODEL
model = Sequential()

# CONVOLUTION LAYERS
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=latticeshape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(3, (3, 3), activation='relu'))

# FLATTEN
model.add(Flatten())

# DENSE LAYERS
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

# OUTPUT LOGIT LAYER
model.add(Dense(2, activation='softmax'))

# DISPLAY NETWORK ARCHITECTURE
model.summary()

# COMPILE MODEL
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# TRAIN MODEL
history = model.fit(x_train, y_train, batch_size=50, epochs=5, verbose=1, shuffle=True, validation_data=(x_test, y_test))

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
print('Test loss:', score[0])
print('Test accuracy:', score[1])

xpredict = np.array(np.load("predictdata.npy"))
xpred = xpredict.reshape(xpredict.shape[0], N, N, 1)

ypred = model.predict_classes(xpred)
ypredict = ypred

ylabels = np.sum(xpred, axis=1)
ylabels = np.abs(ylabels/(N*N))
ylabels [ylabels>=0.5] = 1
ylabels[ylabels<0.5] = 0

yerror = np.abs(ypredict-ylabels)

order = np.abs(np.sum([xpredict], axis=2))
order = order.reshape(order.shape[1],)
order = order/1024
plt.plot(ypredict,order,'x')
