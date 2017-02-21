import os
import time
import cv2
import numpy as np

from scipy import misc

from matplotlib import pyplot as plt
from Helper_Tools import load_reference_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin,\
    save_rois_with_label
from Video_Tools import load_video, get_video_dimensions, normalize_mean_over_interval, split_frame_into_rgb_channels


# input_dir_path = os.path.join('assets', 'Vid_Original')
input_dir_path = os.path.join('Neural_Net', 'assets')
dest_dir_path = os.path.join('assets', 'Pulse_Data', '')
dest_skin_dir_path = os.path.join('assets', 'Skin_Label_Data', '')
signal_data_dir = os.path.join('Neural_Net', 'assets', 'Pulse_Data')
skin_mask_data_dir = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data')
out_balanced_dir = os.path.join('Neural_Net', 'assets', 'Balanced_Data')

file = 'Balanced_00130.npy'

# file_path_1 = os.path.join(out_balanced_dir, file)
# file_path_2 = os.path.join(input_dir_path, file)
#
# data_1 = np.load(file_path_1)
# data_2 = np.load(file_path_2)
#
#
# print(np.array_equal(data_1, data_2))


print(input_dir_path)

for file in os.listdir(skin_mask_data_dir):
    if file.endswith(".npy"):
        file_path = os.path.join(skin_mask_data_dir, file)

        print(file_path)

        # img = misc.imread(file_path)

        mean_img = np.load(file_path)
        #
        # mean_img = np.mean(img, axis=2)
        # white_index = mean_img >= 200.0
        #
        #
        skin_index = mean_img == 1.0
        #
        skin_pixel_count = len(mean_img[skin_index])
        # mean_img[white_index] = 0

        fig = plt.figure(figsize=(17, 9))
        sub1 = fig.add_subplot(111)
        sub1.set_title('Skin Pixel: ' + str(skin_pixel_count))
        sub1.imshow(mean_img)
        # fig.savefig('assets\\Skin_00149.jpg')

        print(np.amin(mean_img))
        print(np.amax(mean_img))

        # np.save('Skin_00149.npy', mean_img)

        plt.show()



