# How to load and use weights from a checkpoint
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Convolution2D, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import numpy as np
import os


# fix random seed for reproducibility
from keras.utils import np_utils


seed = 7
np.random.seed(seed)

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

def compile_cnn_model(_weights_path):

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

    cnn.summary()

    cnn.load_weights(_weights_path)
    # cnn_params = cnn.count_params()
    # cnn_config = cnn.get_config()
    # cnn_weights = cnn.get_weights()
    return cnn


if __name__ == '__main__':

    # weights_path = os.path.join('assets', 'CNN_2_Best_Weights_Server.hdf5')
    weights_path = os.path.join('assets', 'CNN_2_Best_Weights.hdf5')

    roi_validation_data_dir = os.path.join('assets', 'Validation_Data', 'ROIs')
    prediction_data_dir = os.path.join('assets', 'Pulse_Data', 'Me')
    # prediction_data_dir = os.path.join('assets', 'Pulse_Data', 'ROIs', 'Me')
    # prediction_data_path = os.path.join('assets', 'Pulse_Data', 'ROIs', 'ROI_00146.npy')

    cnn_model = compile_cnn_model(weights_path)

    for file in os.listdir(prediction_data_dir):
        in_file_path = os.path.join(prediction_data_dir, file)
        if file.endswith(".npy"):

            prediction_data_path = os.path.join(prediction_data_dir, file)
            npy_me = np.load(prediction_data_path)
            npy_me = npy_me.astype('float32')

            prediction_data = npy_me.reshape(npy_me.shape[0]*npy_me.shape[1], npy_me.shape[2], -1, 1)

            prediction = cnn_model.predict(prediction_data)
            pred_img_mask = np.ones((prediction.shape[0], 1))
            skin_ind = prediction[:, 0] >= 0.5
            pred_img_mask[skin_ind] = 0
            # pred_img_mask = pred_img_mask.reshape(32, 64)
            pred_img_mask = pred_img_mask.reshape(1080, 1920)

            plt.figure(figsize=(15, 10))
            plt.imshow(pred_img_mask, cmap=plt.cm.gray)
            plt.suptitle(file, fontsize=18, fontweight='bold')
            out_path = os.path.join('history', 'predict_' + file[:-4] + '.png')
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches='tight')
            print('Saved to ' + out_path)
            plt.show()
            # plt.close()

    cnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', 'fmeasure'])

    '''Evaluation'''
    for file in os.listdir(roi_validation_data_dir):

        valid_data_X_test, valid_data_y_test = get_test_set(os.path.join(roi_validation_data_dir, file))

        scores = cnn_model.evaluate(valid_data_X_test, valid_data_y_test, verbose=0)
        print(file)
        print('IRNN test score:', scores[0])
        print('IRNN test accuracy:', scores[1])
        print('IRNN test fmeasure:', scores[2])

