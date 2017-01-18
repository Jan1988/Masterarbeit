import numpy as np
import os

from sklearn.cross_validation import train_test_split


def get_dataset():

    balanced_signal_data_path = os.path.join('assets', 'Balanced_00130.npy')

    pixel_count = 2073600
    width = 1
    height = 1
    # signal_data = np.ndarray((pixel_count, 44, width, height), dtype=np.float64)

    signal_data = np.load(balanced_signal_data_path)

    X = signal_data[:, 0:44]
    Y = signal_data[:, 44]

    # reshaped_signal_data = signal_data.reshape((pixel_count, 44, width, height))

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # split into 80% for train and 20% for test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

    return X_train, y_train, X_test, y_test
