
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from Neural_Net.Load_Dataset import get_dataset
import numpy


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

pulse_signal_dataset_path = os.path.join('assets', 'ROI_Full_Dataset.npy')


model = Sequential()
# C1
model.add(Convolution2D(5, 11, 1, border_mode="same", activation="relu", input_shape=input_shape))
# S2
model.add(MaxPooling2D(pool_size=(4, 1)))
# C3
model.add(Convolution2D(25, 9, 1, border_mode="same", activation="relu"))
# S4
model.add(MaxPooling2D(pool_size=(4, 1)))
# C5
model.add(Convolution2D(125, 6, 1, border_mode="same", activation="relu"))
# S6
model.add(MaxPooling2D(pool_size=(2, 1)))

# F7
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

# load weights
model.load_weights("weights.best.hdf5")

# define optimizer and objective, compile cnn
model.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['accuracy', 'fmeasure'])


# the data, shuffled and split between train and test sets
X_train, y_train, X_test, y_test = get_dataset(pulse_signal_dataset_path)

# process the data to fit in a keras CNN properly
# input data needs to be (N, X, Y, C) - shaped where
# N - number of samples
# C - number of channels per sample
# (X, Y) - sample size
length_training = X_train.shape[0]
X_train = X_train.reshape(length_training, 44, 1, 1)
X_train = X_train.astype('float32')


scores = model.evaluate(X, Y, verbose=0)

for i, metric in enumerate(scores):
    print(str(i+1) + '. Metric: ' + str(metric))


