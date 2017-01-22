from __future__ import print_function

import numpy as np
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from Neural_Net.Load_Dataset import get_dataset


# pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'Balanced_00160.npy')
pulse_signal_dataset_path = os.path.join('assets', 'ROI_Full_Dataset.npy')

# fix random seed for reproducibility
np.random.seed(7)

batch_size = 128
# nb_classes = 10
nb_epochs = 500
hidden_units = 50

learning_rate = 1e-6
clip_norm = 1.0


# the data, shuffled and split between train and test sets
X_train, y_train, X_test, y_test = get_dataset(pulse_signal_dataset_path)

# Reshape the data to be used by a Tensorflow CNN. Shape is
# (nb_of_samples, img_width, img_heigh, nb_of_color_channels)
X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = y_train.astype('uint8')
y_test = y_test.astype('uint8')
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print('Evaluate IRNN...')
model = Sequential()
model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.001, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='relu',
                    input_shape=X_train.shape[1:]))
model.add(Dense(2))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['fmeasure', 'mean_squared_error'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])


