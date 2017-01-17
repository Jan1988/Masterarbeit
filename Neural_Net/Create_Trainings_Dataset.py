
import os
import pandas as pd
import numpy as np
import time


def prepare_skin_mask(_in_skin_mask_path, _out_skin_mask_path):

    # Read the array in .txt from disk
    new_data = pd.read_csv(_in_skin_mask_path, sep=" ", header=None)

    # original shape of the array
    new_data = new_data.values
    # Reshape it to 1dimensional array
    reshaped_new_data = new_data.reshape((1080 * 1920))

    # Save it as .npy file
    np.save(_out_skin_mask_path, reshaped_new_data)

    # test loading and equality
    npy_loaded = np.load(_out_skin_mask_path)
    print(np.array_equal(reshaped_new_data, npy_loaded))

    one_index = npy_loaded > 0
    print(len(npy_loaded[one_index]))


def pulse_data_txt_to_npy(_in_pulse_data_path, _out_pulse_data_path):
    # Read the array in .txt from disk
    new_data = pd.read_csv(_in_pulse_data_path, sep=" ", header=None)

    # original shape of the array
    new_data = new_data.values

    # Save it as .npy file
    np.save(_out_pulse_data_path, new_data)

    # test loading and equality
    npy_loaded = np.load(_out_pulse_data_path)
    print(np.array_equal(new_data, npy_loaded))


if __name__ == '__main__':
    start_time = time.time()

    # skin_mask_dir = os.path.join('assets', 'Skin_Label_Data')
    # in_skin_mask_file = 'Skin_00130.txt'
    # out_skin_mask_file = 'Skin_00130.npy'
    # in_skin_mask_path = os.path.join(skin_mask_dir, in_skin_mask_file)
    # out_skin_mask_path = os.path.join(skin_mask_dir, out_skin_mask_file)

    pulse_data_dir = os.path.join('assets', 'Pulse_Data')
    in_pulse_data_file = '00130.txt'
    out_pulse_data_file = '00130.npy'
    in_pulse_data_path = os.path.join(pulse_data_dir, in_pulse_data_file)
    out_pulse_data_path = os.path.join(pulse_data_dir, out_pulse_data_file)

    w_div = 16
    h_div = 8

    width = 1920
    height = 1080

    w_steps = int(width / w_div)
    h_steps = int(height / h_div)

    # prepare_skin_mask(in_skin_mask_path, out_skin_mask_path)
    pulse_data_txt_to_npy(in_pulse_data_path, out_pulse_data_path)

# for i, val in enumerate(reshaped_new_data):
#
#     print(npy_loaded[i])
#     print(val)
#     print(val == npy_loaded[i])

# reshaped_new_data = new_data.reshape((1080, 1920, 44))
# print(np.array_equal(npy_loaded, reshaped_new_data))

#
#
# for x in range(0, width, w_steps):
#     for y in range(0, height, h_steps):
#         roi_time_series = video_frames[:, y:y + h_steps, x:x + w_steps]

print("--- Algorithm Completed %s seconds ---" % (time.time() - start_time))