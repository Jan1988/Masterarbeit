
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Convolution2D, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from Neural_Net.Load_Dataset import get_dataset
import numpy as np
import os


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


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
    prediction_data_path = os.path.join('assets', 'ROI_00130.npy')
    # prediction_data_path = os.path.join('assets', 'Pulse_Data', 'ROIs', 'ROI_00146.npy')

    cnn_model = compile_cnn_model(weights_path)

    npy_me = np.load(prediction_data_path)
    npy_me = npy_me.astype('float32')

    prediction_data = npy_me.reshape(npy_me.shape[0]*npy_me.shape[1], npy_me.shape[2], -1, 1)

    prediction = cnn_model.predict(prediction_data)
    pred_img_mask = np.ones((prediction.shape[0], 1))
    skin_ind = prediction[:, 0] <= 0.5
    pred_img_mask[skin_ind] = 0
    pred_img_mask = pred_img_mask.reshape(32, 64)

    plt.imshow(pred_img_mask)
    plt.show()

    print(prediction)



