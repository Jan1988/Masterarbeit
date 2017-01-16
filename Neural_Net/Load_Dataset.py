import numpy as np
import os

from sklearn.cross_validation import train_test_split


def get_dataset():

    signal_data_path = os.path.join('assets', '00128.npy')
    skin_mask_data_path = os.path.join('assets', 'Skin_00128.npy')

    pixel_count = 2073600
    width = 1
    height = 1

    signal_data = np.ndarray((pixel_count, 44, width, height), dtype=np.float64)


    signal_data = np.load(signal_data_path)
    # signal_data[:, :, 1, 1] = loaded_data[:, :]
    reshaped_signal_data = signal_data.reshape((pixel_count, 44, 1, 1))

    skin_mask_data = np.load(skin_mask_data_path)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # split into 80% for train and 20% for test
    X_train, X_test, y_train, y_test = train_test_split(reshaped_signal_data, skin_mask_data, test_size=0.20, random_state=seed)

    return X_train, X_test, y_train, y_test
