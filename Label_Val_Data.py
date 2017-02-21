import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt

start_time = time.time()


# iterate over all files in folder
def multi_valid_data_labeling(_signal_data_dir, _skin_mask_data_dir, out_labeled_data, roi=False):

    for file in os.listdir(_signal_data_dir):

        if file.endswith(".npy"):
            _signal_file_path = os.path.join(_signal_data_dir, file)
            if roi:
                print('ROI Calculation')
                roi_skin_mask_file_path = os.path.join(_skin_mask_data_dir, 'ROI_Skin_' + file[4:])
                single_data_labeling(file, _signal_file_path, roi_skin_mask_file_path, out_labeled_data)
            else:
                print('Pixelwise Calculation')
                _skin_mask_file_path = os.path.join(_skin_mask_data_dir, 'Skin_' + file)
                single_data_labeling(file, _signal_file_path, _skin_mask_file_path, out_labeled_data)


# Sets the Skin-mask of a Pulse signal file as its Labels
def single_data_labeling(_signal_file, _signal_file_path, _skin_mask_file_path, out_labeled_data):

    signal_data = np.load(_signal_file_path)
    print('Load ' + _signal_file_path)
    skin_mask_data = np.load(_skin_mask_file_path)
    print('Load ' + _skin_mask_file_path)

    print(signal_data.shape)
    print(skin_mask_data.shape)

    # Replace remaining NaNs with zero
    where_are_NaNs = np.isnan(signal_data)
    print('NaNs: ' + str(len(signal_data[where_are_NaNs])))
    signal_data[where_are_NaNs] = 0.0

    # skin positions where are ones & non-skin where zeros
    skin_indices = skin_mask_data > 0
    non_skin_indices = skin_mask_data < 1
    skin_count = len(skin_mask_data[skin_indices])
    non_skin_count = len(skin_mask_data[non_skin_indices])

    print('Count of Skin Samples: ' + str(skin_count))
    print('Count of Non-Skin Samples: ' + str(non_skin_count))

    # 2 Arrays with the number of Class Lables as height
    one_labels = np.ones((skin_count, 1))
    zero_labels = np.zeros((non_skin_count, 1))

    # All values at the position of skin
    skin_signal_data = signal_data[skin_indices, :]
    # All values at the position of non-skin
    non_skin_signal_data = signal_data[non_skin_indices, :]

    print(skin_signal_data.shape)
    print(non_skin_signal_data.shape)

    # append right labels to the samplerows
    final_skin_signal_data = np.append(skin_signal_data, one_labels, axis=1)
    final_non_skin_signal_data = np.append(non_skin_signal_data, zero_labels, axis=1)

    print('Skin Samples shape: ' + str(final_skin_signal_data.shape))
    print('Non-Skin Samples shape: ' + str(final_non_skin_signal_data.shape))
    print(np.mean(final_skin_signal_data[:, 44]))
    print(np.mean(final_non_skin_signal_data[:, 44]))

    # concatenate non-skin and skin data lists
    labeled_signal_data = np.concatenate((final_skin_signal_data, final_non_skin_signal_data))
    print(labeled_signal_data.shape)

    labeled_signal_data_path = os.path.join(out_labeled_data, 'Valid_' + _signal_file)

    np.save(labeled_signal_data_path, labeled_signal_data)
    print('Saved to ' + labeled_signal_data_path)


if __name__ == '__main__':

    signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data')
    roi_signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data', 'ROIs')
    me_roi_signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data', 'ROIs', 'Me')
    skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data')
    roi_skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data', 'ROIs')
    me_roi_skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data', 'ROIs', 'Me')

    file = '00130.npy'
    dest_dir_path = os.path.join('Neural_Net', 'assets', 'Validation_Data')
    file_path = os.path.join(signal_data_dir, file)
    skin_mask_file_path = os.path.join(skin_mask_data_dir, 'Skin_' + file)

    # single_data_labeling(file, file_path, skin_mask_file_path, dest_dir_path)
    single_data_labeling(file, file_path, skin_mask_file_path, dest_dir_path)
    # multi_valid_data_labeling(me_roi_signal_data_dir, me_roi_skin_mask_data_dir, dest_dir_path, roi=True)

    print("--- %s seconds ---" % (time.time() - start_time))