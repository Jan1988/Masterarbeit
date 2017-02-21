
import numpy as np
seed = 7
np.random.seed(seed)
import os

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils


# old function for splitting dataset into training- and testset
def split_into_training_and_test(_pulse_signal_dataset, out_dir, _roi=False):

    # seperate samples from labels
    X = _pulse_signal_dataset[:, 0:44]
    Y = _pulse_signal_dataset[:, 44]

    # Dataset Normalization
    mean_X = np.mean(X)
    std_X = np.std(X)
    norm_X = (X-mean_X)/std_X

    # Mean should be near zero and std=1
    print(np.mean(norm_X))
    print(np.std(norm_X))

    X_train, X_test, y_train, y_test = train_test_split(norm_X, Y, test_size=0.20, random_state=seed)

    # process the data to fit in a keras CNN properly
    # input data needs to be (N, X, Y, C) - shaped where
    # N - number of samples
    # C - number of channels per sample
    # (X, Y) - sample size
    length_training = X_train.shape[0]
    length_testing = X_test.shape[0]

    X_train = X_train.reshape(length_training, 44, 1, 1)
    X_test = X_test.reshape(length_testing, 44, 1, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = y_train.astype('uint8')
    y_test = y_test.astype('uint8')
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    print(len(X_train), len(X_test), len(y_train), len(y_test))

    # for saving to files
    out_x_train = 'X_train'
    out_x_test = 'X_test'
    out_y_train = 'y_train'
    out_y_test = 'y_test'

    if _roi:
        out_x_train = 'ROI_' + out_x_train
        out_x_test = 'ROI_' + out_x_test
        out_y_train = 'ROI_' + out_y_train
        out_y_test = 'ROI_' + out_y_test

    out_x_train_path = os.path.join(dataset_dir, out_x_train)
    out_x_test_path = os.path.join(dataset_dir, out_x_test)
    out_y_train_path = os.path.join(dataset_dir, out_y_train)
    out_y_test_path = os.path.join(dataset_dir, out_y_test)

    np.save(out_x_train_path, X_train)
    print('Saving ' + out_x_train_path)
    np.save(out_x_test_path, X_test)
    print('Saving ' + out_x_test_path)
    np.save(out_y_train_path, y_train)
    print('Saving ' + out_y_train_path)
    np.save(out_y_test_path, y_test)
    print('Saving ' + out_y_test_path)


# stack the single balanced pulse data files
def create_full_dataset(_balanced_data_dir, dataset_dir, roi=False):

    # create empty dataset
    full_dataset = np.array([]).reshape(0, 45)

    for file in os.listdir(_balanced_data_dir):

        if file.endswith(".npy") and file[:9] == 'Balanced_':
            balanced_data = os.path.join(_balanced_data_dir, file)

            data_loaded = np.load(balanced_data)

            print('Loaded Shape ' + str(data_loaded.shape))

            # stack loaded data and tempory full_dataset
            full_dataset = np.vstack([full_dataset, data_loaded])

            print('Full Shape ' + str(full_dataset.shape))

    out_file = 'Full_Dataset.npy'
    if roi:
        out_file = 'ROI_' + out_file

    out_full_dataset_path = os.path.join(dataset_dir, out_file)
    np.save(out_full_dataset_path, full_dataset)
    print('Saving ' + out_full_dataset_path)

    # # call function to get a splitted dataset
    # split_into_training_and_test(full_dataset, dataset_dir, roi)


if __name__ == '__main__':

    file = '00130.npy'
    # Input
    balanced_data_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data')
    roi_balanced_data_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data', 'ROIs')
    # Destination folder
    dataset_dir = os.path.join('Neural_Net', 'assets', 'Datasets')

    # create_full_dataset(roi_balanced_data_dir, dataset_dir, roi=True)
    create_full_dataset(balanced_data_dir, dataset_dir)
