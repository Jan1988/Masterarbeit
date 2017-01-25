from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
import os
from matplotlib import pyplot as plt
# fix random seed for reproducibility

import tensorflow as tf


from Neural_Net.Load_Dataset import get_dataset


def plot_acc_and_loss():

    # list all data in history
    print(history.history.keys())

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(fontsize=20, fontweight='bold')

    # summarize history for accuracy
    sub1 = fig.add_subplot(121)
    sub1.plot(history.history['acc'])
    sub1.plot(history.history['val_acc'])
    sub1.title('model accuracy')
    sub1.ylabel('accuracy')
    sub1.xlabel('epoch')
    sub1.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    sub2 = fig.add_subplot(121)
    sub2.plot(history.history['loss'])
    sub2.plot(history.history['val_loss'])
    sub2.title('model loss')
    sub2.ylabel('loss')
    sub2.xlabel('epoch')
    sub2.legend(['train', 'test'], loc='upper left')

    plt.show()


def compile_cnn_model(_input_shape):

    cnn = Sequential()
    # C1
    cnn.add(Convolution2D(5, 11, 1, border_mode="same", input_shape=_input_shape))
    cnn.add(LeakyReLU(alpha=0.3))
    cnn.add(Dropout(0.2))
    # S2
    # cnn.add(MaxPooling2D(pool_size=(4, 1)))
    # C3
    cnn.add(Convolution2D(25, 9, 1, border_mode="same"))
    cnn.add(LeakyReLU(alpha=0.3))
    cnn.add(Dropout(0.2))
    # S4
    cnn.add(MaxPooling2D(pool_size=(4, 1)))
    # C5
    cnn.add(Convolution2D(125, 6, 1, border_mode="same"))
    cnn.add(LeakyReLU(alpha=0.3))
    cnn.add(Dropout(0.2))
    # S6
    cnn.add(MaxPooling2D(pool_size=(2, 1)))
    # F7
    cnn.add(Flatten())
    cnn.add(Dense(512))
    cnn.add(LeakyReLU(alpha=0.3))
    cnn.add(Dropout(0.75))
    cnn.add(Dense(2, activation="softmax"))

    # define optimizer and objective, compile cnn
    cnn.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['accuracy', 'fmeasure'])

    # checkpoint
    checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # cnn.summary()
    # cnn_params = cnn.count_params()
    # cnn_config = cnn.get_config()
    # cnn_weights = cnn.get_weights()

    return cnn, callbacks_list



if __name__ == '__main__':

    tf.set_random_seed(7)
    # pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'Balanced_00130.npy')
    # pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'ROIs', 'Balanced_ROI_00132.npy')
    pulse_signal_dataset_path = os.path.join('assets', 'ROI_Full_Dataset.npy')
    weights_path = os.path.join('assets', 'CNN_2_Best_Weights.hdf5')

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

    # convert class vectors to binary class matrices
    y_train = y_train.astype('uint8')
    y_test = y_test.astype('uint8')
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # input shape for tf: (rows, cols, channels)
    input_shape = (44, 1, 1)
    epochs = 1000
    batch_size = 256




    # for i in range(3):

    model, callbacks_list = compile_cnn_model(input_shape)

    # training
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, nb_epoch=epochs, show_accuracy=True, callbacks=callbacks_list)
    # evaluation
    scores = model.evaluate(X_test, y_test, verbose=0)
    for i, metric in enumerate(scores):
        print(str(i+1) + '. Metric: ' + str(metric))







