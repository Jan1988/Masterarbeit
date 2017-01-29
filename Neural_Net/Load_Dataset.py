import numpy as np


from keras.utils import np_utils
from sklearn.cross_validation import train_test_split


def get_dataset(_dataset_path):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    pulse_signal_dataset = np.load(_dataset_path)

    X = pulse_signal_dataset[:, 0:44]
    Y = pulse_signal_dataset[:, 44]

    # reshaped_signal_data = signal_data.reshape((pixel_count, 44, width, height))

    # split into 80% for train and 20% for test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
    # the data, shuffled and split between train and test sets

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
