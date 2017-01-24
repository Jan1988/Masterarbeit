
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Convolution2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from Neural_Net.Load_Dataset import get_dataset
import numpy as np
import os


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def compile_cnn_model(_weights_path):
    input_shape = (44, 1, 1)

    model = Sequential()
    # C1
    model.add(Convolution2D(5, 8, 1, border_mode="same", activation="relu", input_shape=input_shape))
    # S2
    model.add(MaxPooling2D(pool_size=(4, 1)))
    # C3
    model.add(Convolution2D(10, 6, 1, border_mode="same", activation="relu"))
    # S4
    model.add(MaxPooling2D(pool_size=(4, 1)))
    # C5
    model.add(Convolution2D(25, 5, 1, border_mode="same", activation="relu"))
    # S6
    model.add(MaxPooling2D(pool_size=(2, 1)))
    # F7
    model.add(Flatten())
    model.add(Dense(250, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    # load weights
    model.load_weights(_weights_path)
    # define optimizer and objective, compile cnn
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', 'fmeasure'])

    return model


if __name__ == '__main__':

    # weights_path = os.path.join('assets', 'CNN_2_Best_Weights_Server.hdf5')
    weights_path = os.path.join('assets', 'CNN_2_Best_Weights.hdf5')
    prediction_data_path = os.path.join('assets', 'Predict_ROI_00101.npy')
    # prediction_data_path = os.path.join('assets', 'Pulse_Data', 'ROIs', 'ROI_00146.npy')

    cnn_model = compile_cnn_model(weights_path)


    input_data = np.load(prediction_data_path)
    prediction_data = input_data.reshape(input_data.shape[0]*input_data.shape[1], input_data.shape[2], -1, 1)

    prediction = cnn_model.predict(prediction_data)
    pred_img_mask = np.ones((prediction.shape[0], 1))
    skin_ind = prediction[:, 0] <= 0.5
    pred_img_mask[skin_ind] = 0
    pred_img_mask = pred_img_mask.reshape(32, 64)


    plt.imshow(pred_img_mask)
    plt.show()

    print(prediction)



