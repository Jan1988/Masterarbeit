
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Convolution2D, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import numpy as np
import os


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def predict_multi_npys(_prediction_data_dir):

    for file in os.listdir(_prediction_data_dir):
        if file.endswith(".npy"):

            predict_single_npy(_prediction_data_dir, file)


def predict_single_npy(_prediction_data_dir, file):

    prediction_data_path = os.path.join(_prediction_data_dir, file)
    npy_to_predict = np.load(prediction_data_path)

    npy_to_predict = npy_to_predict.astype('float32')
    prediction_data = npy_to_predict.reshape(npy_to_predict.shape[0] * npy_to_predict.shape[1], npy_to_predict.shape[2], -1, 1)

    prediction = cnn_model.predict(prediction_data)
    pred_img_mask = np.ones((prediction.shape[0], 1))
    skin_ind = prediction[:, 0] <= 0.5
    pred_img_mask[skin_ind] = 0
    pred_img_mask = pred_img_mask.reshape(npy_to_predict.shape[0], npy_to_predict.shape[1])

    plt.imshow(pred_img_mask)
    out_path = os.path.join('history', 'predict_' + file[:-4] + '.png')
    plt.savefig(out_path, bbox_inches='tight')
    print('Saved to ' + out_path)
    plt.close()


def compile_cnn_model(_weights_path):

    # input shape for tf: (rows, cols, channels)
    input_shape = (44, 1, 1)

    cnn = Sequential()
    # C1
    cnn.add(Convolution2D(5, 11, 1, border_mode="same", input_shape=input_shape))
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

    cnn.summary()

    cnn.load_weights(_weights_path)
    # cnn_params = cnn.count_params()
    # cnn_config = cnn.get_config()
    # cnn_weights = cnn.get_weights()
    return cnn


if __name__ == '__main__':

    # weights_path = os.path.join('assets', 'CNN_2_Best_Weights_Server.hdf5')
    weights_path = os.path.join('assets', 'CNN_2_Best_Weights.hdf5')
    prediction_data_dir = os.path.join('assets', 'Pulse_Data', 'Me')
    file_to_predict = 'me_00072.npy'

    cnn_model = compile_cnn_model(weights_path)

    # predict_multi_npys(prediction_data_dir)
    predict_single_npy(prediction_data_dir, file_to_predict)





