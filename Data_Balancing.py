import numpy as np
import os
from random import randrange
import sklearn


def multi_npy_data_balancing(_signal_data_dir, skin_mask_data_dir, out_balanced_signal_data):


    for file in os.listdir(_signal_data_dir):
        # skin_mask_data_file_path = os.path.join(skin_mask_data_dir, 'Skin_' + file)
        roi_skin_mask_data_file_path = os.path.join(skin_mask_data_dir, 'ROI_Skin_' + file[4:])
        skin_mask_exists = os.path.isfile(roi_skin_mask_data_file_path)

        print(file)
        print('Skin mask exists: ' + str(skin_mask_exists))
        if file.endswith(".npy") and skin_mask_exists:
            signal_data_file_path = os.path.join(_signal_data_dir, file)
            out_file_path = os.path.join(out_balanced_signal_data, 'Balanced_' + file)

            # signal_data = np.ndarray((pixel_count, 44, width, height), dtype=np.float64)

            signal_data = np.load(signal_data_file_path)
            skin_mask_data = np.load(roi_skin_mask_data_file_path)

            print(signal_data.shape)
            print(skin_mask_data.shape)

            # Where values are low
            skin_indices = skin_mask_data > 0
            non_skin_indices = skin_mask_data < 1
            skin_count = len(skin_mask_data[skin_indices])
            non_skin_count = len(skin_mask_data[non_skin_indices])

            print('Count of Skin Samples: ' + str(skin_count))
            print('Count of Non-Skin Samples: ' + str(non_skin_count))

            one_labels = np.ones((skin_count, 1))
            zero_labels = np.zeros((skin_count, 1))

            skin_signal_data = signal_data[skin_indices, :]
            non_skin_signal_data = signal_data[non_skin_indices, :]

            print(skin_signal_data.shape)
            print(non_skin_signal_data.shape)

            random_choice = np.random.choice(non_skin_count, size=skin_count, replace=False)

            subsampled_non_skin_signal_data = non_skin_signal_data[random_choice, :]

            final_skin_signal_data = np.concatenate((skin_signal_data, one_labels), axis=1)
            final_non_skin_signal_data = np.concatenate((subsampled_non_skin_signal_data, zero_labels), axis=1)

            print('Skin Samples shape: ' + str(final_skin_signal_data.shape))
            print('Non-Skin Samples shape: ' + str(final_non_skin_signal_data.shape))

            balanced_signal_data = np.concatenate((final_skin_signal_data, final_non_skin_signal_data))

            print(balanced_signal_data.shape)

            print(np.amin(balanced_signal_data[:, 44]))
            print(np.amax(balanced_signal_data[:, 44]))
            print(np.amin(balanced_signal_data))
            print(np.amax(balanced_signal_data))

            print('Saving: ' + out_file_path)
            np.save(out_file_path, balanced_signal_data)




if __name__ == '__main__':

    file = '00130.npy'
    signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data')
    roi_signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data', 'ROIs')
    skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data')
    roi_skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data', 'ROIs')
    out_balanced_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data')
    out_roi_balanced_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data', 'ROIs')

    multi_npy_data_balancing(roi_signal_data_dir, roi_skin_mask_data_dir, out_roi_balanced_dir)
