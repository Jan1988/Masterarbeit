
import os
import pandas as pd
import numpy as np
import time


start_time = time.time()

input_dir_path = os.path.join('assets', 'Pulse_Data')
skin_mask_dir_path = os.path.join('assets', 'Skin_Label_Data', 'Gute_Ergebnisse')

file = '00128.txt'
skin_mask_file = 'Skin_00128.txt'
file_path = os.path.join(input_dir_path, file)
skin_mask_file_path = os.path.join(skin_mask_dir_path, skin_mask_file)

w_div = 16
h_div = 8

width = 1920
height = 1080

w_steps = int(width / w_div)
h_steps = int(height / h_div)

# Read the array from disk
new_data = pd.read_csv(skin_mask_file_path, sep=" ", header=None)

# original shape of the array
new_data = new_data.values
#

reshaped_new_data = new_data.reshape((1080*1920))

np.save('Skin_00128.npy', reshaped_new_data)


npy_loaded = np.load('Skin_00128.npy')
print(np.array_equal(reshaped_new_data, npy_loaded))



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