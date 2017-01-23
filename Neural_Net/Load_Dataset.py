import numpy as np
import os

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


    return X_train, y_train, X_test, y_test
