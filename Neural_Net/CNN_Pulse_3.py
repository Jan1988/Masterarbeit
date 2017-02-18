import cv2
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
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Activation, BatchNormalization
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


def plot_acc_and_loss(_history_callback):

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


def compile_cnn_model():

    # input shape for tf: (rows, cols, channels)
    input_shape = (44, 1, 1)

    cnn = Sequential()
    # C1
    cnn.add(Convolution2D(5, 11, 1, border_mode="same", input_shape=input_shape))
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))

    # S2
    cnn.add(MaxPooling2D(pool_size=(2, 1)))
    # C3
    cnn.add(Convolution2D(10, 9, 1, border_mode="same"))
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))

    # S4
    cnn.add(MaxPooling2D(pool_size=(2, 1)))
    # C5
    cnn.add(Convolution2D(25, 6, 1, border_mode="same"))
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    # S6
    cnn.add(MaxPooling2D(pool_size=(4, 1)))
    # F7
    cnn.add(Flatten())
    cnn.add(Dense(125))
    cnn.add(BatchNormalization())
    # cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.5))

    cnn.add(Dense(2))
    cnn.add(BatchNormalization())
    cnn.add(Activation('softmax'))

    # define optimizer and objective, compile cnn
    cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', 'fmeasure'])

    cnn.summary()
    # cnn_params = cnn.count_params()
    # cnn_config = cnn.get_config()
    # cnn_weights = cnn.get_weights()

    return cnn


def get_test_set(_validation_data_path):

    validation_data = np.load(_validation_data_path)

    X_test = validation_data[:, 0:44]
    y_test = validation_data[:, 44]

    length_testing = X_test.shape[0]
    X_test = X_test.reshape(length_testing, 44, 1, 1)
    X_test = X_test.astype('float32')
    y_test = y_test.astype('uint8')
    y_test = np_utils.to_categorical(y_test)

    return X_test, y_test


def get_trainings_set(_dataset_path, ):

    trainings_dataset = np.load(_dataset_path)

    X_train = trainings_dataset[:, 0:44]
    y_train = trainings_dataset[:, 44]

    # # Dataset Normalization
    # print(np.amin(X))
    # print(np.amax(X))
    # mean_X = np.mean(X)
    # std_X = np.std(X)
    # norm_X = (X-mean_X)/std_X
    # print(np.amin(norm_X))
    # print(np.amax(norm_X))
    # print(np.mean(norm_X))
    # print(np.std(norm_X))

    # # Dataset Normalization

    print(X_train.shape, y_train.shape)

    # process the data to fit in a keras CNN properly
    # input data needs to be (N, X, Y, C) - shaped where
    # N - number of samples
    # C - number of channels per sample
    # (X, Y) - sample size
    length_training = X_train.shape[0]

    X_train = X_train.reshape(length_training, 44, 1, 1)
    X_train = X_train.astype('float32')

    # convert class vectors to binary class matrices
    y_train = y_train.astype('uint8')
    y_train = np_utils.to_categorical(y_train)

    print(X_train.shape, y_train.shape)

    return X_train, y_train


if __name__ == '__main__':
    start_time = time.time()

    dataset_dir = os.path.join('assets', 'Datasets')
    # validation_data_dir = os.path.join('assets', 'Validation_Data')
    roi_validation_data_dir = os.path.join('assets', 'Validation_Data', 'ROIs')
    weights_path = os.path.join('assets', 'CNN_2_Best_Weights.hdf5')

    dataset_path = os.path.join(dataset_dir, 'ROI_Full_Dataset.npy')
    validation_data_path = os.path.join(roi_validation_data_dir, 'Valid_ROI_00076.npy')

    X_train, y_train = get_trainings_set(dataset_path)
    X_test, y_test = get_test_set(validation_data_path)

    epochs = 200
    batch_size = 1024

    # for i in range(3):
    # checkpoint
    checkpoint = ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model = compile_cnn_model()

    # training
    history_callback = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, nb_epoch=epochs, show_accuracy=True, callbacks=callbacks_list)
    print("--- Training Completed %s seconds ---" % (time.time() - start_time))



    '''Evaluation'''
    for file in os.listdir(roi_validation_data_dir):

        valid_data_X_test, valid_data_y_test = get_test_set(os.path.join(roi_validation_data_dir, file))

        scores = model.evaluate(valid_data_X_test, valid_data_y_test, verbose=0)
        print(file)
        print('IRNN test score:', scores[0])
        print('IRNN test accuracy:', scores[1])
        print('IRNN test fmeasure:', scores[2])


    prediction_data_dir = os.path.join('assets', 'Pulse_Data', 'ROIs', 'Me')
    for file_to_predict in os.listdir(prediction_data_dir):

        prediction_data_path = os.path.join(prediction_data_dir, file_to_predict)
        npy_to_predict = np.load(prediction_data_path)

        npy_to_predict = npy_to_predict.astype('float32')
        prediction_data = npy_to_predict.reshape(npy_to_predict.shape[0] * npy_to_predict.shape[1],
                                                 npy_to_predict.shape[2], -1, 1)

        prediction = model.predict(prediction_data)
        pred_img_mask = np.ones((prediction.shape[0], 1))
        skin_ind = prediction[:, 0] >= 0.5
        pred_img_mask[skin_ind] = 0
        pred_img_mask = pred_img_mask.reshape(npy_to_predict.shape[0], npy_to_predict.shape[1])


        plt.figure(figsize=(15, 10))
        plt.imshow(pred_img_mask, cmap=plt.cm.gray)
        plt.suptitle(file_to_predict, fontsize=18, fontweight='bold')
        # out_path = os.path.join('history', 'predict_' + file[:-4] + '.png')
        # plt.savefig(out_path, bbox_inches='tight')
        # print('Saved to ' + out_path)
        plt.tight_layout()
        plt.show()



    plot_acc_and_loss(history_callback)





