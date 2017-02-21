import numpy as np
import os
from random import randrange
import sklearn


balanced_data_count = 0

# iterate over all files in folder
def multi_npy_data_balancing(_signal_data_dir, _skin_mask_data_dir, _out_balanced_dir, roi=False):

    for file in os.listdir(_signal_data_dir):

        if file.endswith(".npy"):
            _signal_file_path = os.path.join(_signal_data_dir, file)
            if roi:
                print('ROI Calculation')
                roi_skin_mask_file_path = os.path.join(_skin_mask_data_dir, 'ROI_Skin_' + file[4:])
                single_npy_data_balancing(file, _signal_file_path, roi_skin_mask_file_path, _out_balanced_dir)
            else:
                print('Per Pixelwise Calculation')
                _skin_mask_file_path = os.path.join(_skin_mask_data_dir, 'Skin_' + file)
                single_npy_data_balancing(file, _signal_file_path, _skin_mask_file_path, _out_balanced_dir)


# applying skin-mask to set a class label for every sample
# subsampling of non-skin class samples for balancing data
def single_npy_data_balancing(_signal_file, _signal_file_path, _skin_mask_file_path, _out_balanced_dir):

    out_file_path = os.path.join(_out_balanced_dir, 'Balanced_' + _signal_file)

    # load mask and pulse data of one video file
    signal_data = np.load(_signal_file_path)
    skin_mask_data = np.load(_skin_mask_file_path)

    print(signal_data.shape)
    print(skin_mask_data.shape)

    # Replace remaining NaNs with zero
    # there were still some left
    where_are_NaNs = np.isnan(signal_data)
    print('NaNs: ' + str(len(signal_data[where_are_NaNs])))
    signal_data[where_are_NaNs] = 0.0

    # Get indices of skin pixels
    skin_indices = skin_mask_data > 0
    # Get indices of non-skin pixels
    non_skin_indices = skin_mask_data < 1

    skin_count = len(skin_mask_data[skin_indices])
    non_skin_count = len(skin_mask_data[non_skin_indices])

    print('Count of Skin Samples: ' + str(skin_count))
    print('Count of Non-Skin Samples: ' + str(non_skin_count))

    # 2 Arrays with the number of Class Lables as height
    one_labels = np.ones((skin_count, 1))
    zero_labels = np.zeros((skin_count, 1))

    # All values at the position of skin
    skin_signal_data = signal_data[skin_indices, :]
    # All values at the position of non-skin
    non_skin_signal_data = signal_data[non_skin_indices, :]

    print(skin_signal_data.shape)
    print(non_skin_signal_data.shape)

    # number of non-skin data-rows should be the same as skin data-rows
    random_choice = np.random.choice(non_skin_count, size=skin_count, replace=False)
    subsampled_non_skin_signal_data = non_skin_signal_data[random_choice, :]

    # Concatenate right labels to the samplerows
    final_skin_signal_data = np.concatenate((skin_signal_data, one_labels), axis=1)
    final_non_skin_signal_data = np.concatenate((subsampled_non_skin_signal_data, zero_labels), axis=1)

    print('Skin Samples shape: ' + str(final_skin_signal_data.shape))
    print('Non-Skin Samples shape: ' + str(final_non_skin_signal_data.shape))

    # concatenate non-skin and skin data lists
    balanced_signal_data = np.concatenate((final_skin_signal_data, final_non_skin_signal_data))

    print(balanced_signal_data.shape)

    print(np.amin(balanced_signal_data[:, 44]))
    print(np.amax(balanced_signal_data[:, 44]))
    print('Min: ' + str(np.amin(balanced_signal_data)))
    print('Mean: ' + str(np.mean(balanced_signal_data)))
    print('Max: ' + str(np.amax(balanced_signal_data)))

    print('Saving: ' + out_file_path)
    np.save(out_file_path, balanced_signal_data)

    # to see how many samples are processed
    global balanced_data_count
    balanced_data_count += balanced_signal_data.shape[0]


if __name__ == '__main__':

    signal_file = 'ROI_00132.npy'
    skin_mask_file = 'ROI_Skin_00132.npy'

    signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data')
    roi_signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data', 'ROIs')
    skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data')
    roi_skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data', 'ROIs')
    out_balanced_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data')
    out_roi_balanced_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data', 'ROIs')

    #
    signal_file_path = os.path.join(roi_signal_data_dir, signal_file)
    skin_mask_file_path = os.path.join(roi_skin_mask_data_dir, skin_mask_file)

    #
    # multi_npy_data_balancing(roi_signal_data_dir, roi_skin_mask_data_dir, out_roi_balanced_dir, roi=True)
    multi_npy_data_balancing(signal_data_dir, skin_mask_data_dir, out_balanced_dir)
    # single_npy_data_balancing(signal_file, signal_file_path, skin_mask_file_path, out_balanced_dir)
    global balanced_data_count

    print('Total samples balanced: ' + str(balanced_data_count))
