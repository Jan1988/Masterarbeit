import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import tensorflow as tf
tf.set_random_seed(7)
import time

from sklearn.cross_validation import train_test_split
# tf.set_random_seed(7)

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
import os
import numpy as np
from matplotlib import pyplot as plt

# from Neural_Net.Load_Dataset import get_dataset
from Load_Dataset import get_dataset

# def mean_pred(y_true, y_pred):
#
#     # _metric = y_pred/y_true
#
#     return _metric

def plot_acc_and_loss(_model, _history_callback):

    # list all data in history
    print(_history_callback.history.keys())

    np_val_loss_history = np.array(history_callback.history['val_loss'])
    np_val_acc_history = np.array(history_callback.history['val_acc'])

    val_loss_path = os.path.join('history', 'val_loss_history.txt')
    val_acc_path = os.path.join('history', 'val_acc_history.txt')

    np.savetxt(val_loss_path, np_val_loss_history, delimiter=",")
    np.savetxt(val_acc_path, np_val_acc_history, delimiter=",")

    fig = plt.figure(figsize=(12, 8))

    # summarize history for accuracy
    sub1 = fig.add_subplot(121)
    sub1.plot(_history_callback.history['acc'])
    sub1.plot(_history_callback.history['val_acc'])
    sub1.set_title('model accuracy')
    sub1.set_ylabel('accuracy')
    sub1.set_xlabel('epoch')
    sub1.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    sub2 = fig.add_subplot(122)
    sub2.plot(_history_callback.history['loss'])
    sub2.plot(_history_callback.history['val_loss'])
    sub2.set_title('model loss')
    sub2.set_ylabel('loss')
    sub2.set_xlabel('epoch')
    sub2.legend(['train', 'test'], loc='upper left')

    fig.savefig(os.path.join('history', 'Acc_And_Loss.png'), bbox_inches='tight')
    plt.close()
    # plt.show()


def get_dataset(_dataset_path):

    pulse_signal_dataset = np.load(_dataset_path)

    X = pulse_signal_dataset[:, 0:44]
    Y = pulse_signal_dataset[:, 44]

    # Dataset Normalization
    print(np.amin(X))
    print(np.amax(X))
    mean_X = np.mean(X)
    std_X = np.std(X)
    norm_X = (X-mean_X)/std_X
    print(np.amin(norm_X))
    print(np.amax(norm_X))
    print(np.mean(norm_X))
    print(np.std(norm_X))
    # reshaped_signal_data = signal_data.reshape((pixel_count, 44, width, height))

    # # split into 80% for train and 20% for test
    X_train, X_test, y_train, y_test = train_test_split(norm_X, Y, test_size=0.20, random_state=seed)
    # # the data, shuffled and split between train and test sets

    # process the data to fit in a keras CNN properly
    # input data needs to be (N, X, Y, C) - shaped where
    # N - number of samples
    # C - number of channels per sample
    # (X, Y) - sample size
    length_training = X_train.shape[0]
    length_testing = X_test.shape[0]

    print(length_training, length_testing, len(y_train), len(y_test))

    X_train = X_train.reshape(length_training, 44, 1, 1)
    X_test = X_test.reshape(length_testing, 44, 1, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = y_train.astype('uint8')
    y_test = y_test.astype('uint8')
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)


    return X_train, y_train, X_test, y_test


def compile_cnn_model(_input_shape):

    cnn = Sequential()
    # C1
    cnn.add(Convolution2D(5, 11, 1, border_mode="same", input_shape=_input_shape))
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(Activation('relu'))

    # S2
    cnn.add(MaxPooling2D(pool_size=(2, 1)))
    # C3
    cnn.add(Convolution2D(25, 9, 1, border_mode="same"))
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(Activation('relu'))

    # S4
    cnn.add(MaxPooling2D(pool_size=(2, 1)))
    # C5
    cnn.add(Convolution2D(125, 6, 1, border_mode="same"))
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    # S6
    cnn.add(MaxPooling2D(pool_size=(4, 1)))
    # F7
    cnn.add(Flatten())
    cnn.add(Dense(125))
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(2, activation="softmax"))

    # define optimizer and objective, compile cnn
    cnn.compile(loss="binary_crossentropy", optimizer="adam",  metrics=['accuracy', 'fmeasure'])

    cnn.summary()
    # cnn_params = cnn.count_params()
    # cnn_config = cnn.get_config()
    # cnn_weights = cnn.get_weights()

    return cnn


if __name__ == '__main__':

    start_time = time.time()

    # pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'Balanced_00130.npy')
    # pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'ROIs', 'Balanced_ROI_00132.npy')
    pulse_signal_dataset_path = os.path.join('assets', 'ROI_Full_Dataset.npy')
    weights_path = os.path.join('assets', 'CNN_2_Best_Weights.hdf5')

    X_train, y_train, X_test, y_test = get_dataset(pulse_signal_dataset_path)

    # input shape for tf: (rows, cols, channels)
    input_shape = (44, 1, 1)
    epochs = 20
    batch_size = 256

    # for i in range(3):
    # checkpoint
    checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model = compile_cnn_model(input_shape)

    # training
    history_callback = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, nb_epoch=epochs, show_accuracy=True, callbacks=callbacks_list)


    '''Evaluation'''
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('IRNN test score:', scores[0])
    print('IRNN test accuracy:', scores[1])

    # npy_me = np.load('assets/me_2.npy')
    # npy_me = npy_me.astype('float32')
    #
    # prediction_data = npy_me.reshape(npy_me.shape[0]*npy_me.shape[1], npy_me.shape[2], -1, 1)
    #
    # prediction = model.predict(prediction_data)
    # pred_img_mask = np.ones((prediction.shape[0], 1))
    # skin_ind = prediction[:, 0] <= 0.5
    # pred_img_mask[skin_ind] = 0
    # pred_img_mask = pred_img_mask.reshape(32, 64)
    #
    # plt.imshow(pred_img_mask)
    # plt.show()
    #
    # print("--- Algorithm Completed %s seconds ---" % (time.time() - start_time))
    #
    # plot_acc_and_loss(model, history_callback)





