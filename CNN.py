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
y = np.round(label*10)

test_x = test_dataset
test_y = np.round(tlabels*10)

# PREPARING DATA: RESHAPE LATTICE TO N*N*1, COVERT LABELS TO CATEGORICAL

latticeshape = (N, N, 1)
x_train = x.reshape(x.shape[0], N, N, 1)
x_test = test_x.reshape(test_x.shape[0],N,N,1)
y_train = keras.utils.to_categorical(y)
y_test = keras.utils.to_categorical(test_y)

# CREATE MODEL
model = Sequential()

# CONVOLUTION LAYERS
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=latticeshape))
model.add(Conv2D(64, (2, 2), activation='relu'))

# MAX POOL LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))

# DROPOUT AND FLATTEN
model.add(Dropout(0.2))
model.add(Flatten())

# DENSE LAYERS
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))

# OUTPUT LOGIT LAYER
model.add(Dense(36, activation='softmax'))

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
y_predictions = ypred/10

truelabels = np.array(np.load("predictlabels.npy"))
ylabels = np.round(truelabels)