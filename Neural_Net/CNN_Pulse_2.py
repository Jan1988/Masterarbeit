from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
import os
from matplotlib import pyplot as plt
# fix random seed for reproducibility
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)


from Neural_Net.Load_Dataset import get_dataset


# pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'Balanced_00130.npy')
pulse_signal_dataset_path = os.path.join('assets', 'ROI_Full_Dataset.npy')


# input shape for tf: (rows, cols, channels)
input_shape = (44, 1, 1)
epochs = 20
batch_size = 128


# the data, shuffled and split between train and test sets
X_train, y_train, X_test, y_test = get_dataset(pulse_signal_dataset_path)

# process the data to fit in a keras CNN properly
# input data needs to be (N, X, Y, C) - shaped where
# N - number of samples
# C - number of channels per sample
# (X, Y) - sample size
length_training = X_train.shape[0]
length_testing = X_test.shape[0]


X_train = X_train.reshape(length_training, 44, 1, 1)
X_test = X_test.reshape(length_testing, 44, 1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# output labels should be one-hot vectors - ie,
# 0 -> [0, 0, 1]
# 1 -> [0, 1, 0]
# 2 -> [1, 0, 0]

# convert class vectors to binary class matrices
y_train = y_train.astype('uint8')
y_test = y_test.astype('uint8')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# define a CNN
# see http://keras.io for API reference

cnn = Sequential()

# C1
cnn.add(Convolution2D(5, 11, 1, border_mode="same", activation="relu", input_shape=input_shape))

# S2
cnn.add(MaxPooling2D(pool_size=(4, 1)))

# C3
cnn.add(Convolution2D(25, 9, 1, border_mode="same", activation="relu"))

# S4
cnn.add(MaxPooling2D(pool_size=(4, 1)))

# C5
cnn.add(Convolution2D(125, 6, 1, border_mode="same", activation="relu"))

# S6
cnn.add(MaxPooling2D(pool_size=(2, 1)))

# F7
cnn.add(Flatten())
cnn.add(Dense(500, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(2, activation="softmax"))

# define optimizer and objective, compile cnn
cnn.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['accuracy', 'fmeasure'])

# checkpoint
weights_file = "weights.best.hdf5"
checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

cnn.summary()

cnn_params = cnn.count_params()
cnn_config = cnn.get_config()
cnn_weights = cnn.get_weights()


# train
history = cnn.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, nb_epoch=epochs, show_accuracy=True, callbacks=callbacks_list)

scores = cnn.evaluate(X_test, y_test, verbose=0)

for i, metric in enumerate(scores):
    print(str(i+1) + '. Metric: ' + str(metric))

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
