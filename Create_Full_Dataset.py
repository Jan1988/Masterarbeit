
import numpy as np
import os


def create_full_dataset(_balanced_data_dir, _out_full_dataset_path):


    full_dataset = np.array([]).reshape(0, 45)

    for file in os.listdir(_balanced_data_dir):

        if file.endswith(".npy") and file[:9] == 'Balanced_':
            balanced_data = os.path.join(_balanced_data_dir, file)

            data_loaded = np.load(balanced_data)

            print('Loaded Shape ' + str(data_loaded.shape))

            full_dataset = np.vstack([full_dataset, data_loaded])

            print('Full Shape ' + str(full_dataset.shape))

    np.save(_out_full_dataset_path, full_dataset)
    print('Saving ' + _out_full_dataset_path)

if __name__ == '__main__':

    file = '00130.npy'
    balanced_data_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data')
    out_full_dataset_path = os.path.join('Neural_Net', 'assets', 'Full_Dataset.npy')
    roi_balanced_data_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data', 'ROIs')
    roi_out_full_dataset_path = os.path.join('Neural_Net', 'assets', 'ROI_Full_Dataset.npy')

    create_full_dataset(roi_balanced_data_dir, roi_out_full_dataset_path)
