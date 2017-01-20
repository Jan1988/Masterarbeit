import os
import time
import cv2
import numpy as np

from scipy import misc

from matplotlib import pyplot as plt
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin,\
    save_rois_with_label
from Video_Tools import load_video, get_video_dimensions, normalize_mean_over_interval, split_frame_into_rgb_channels


# input_dir_path = os.path.join('assets', 'Vid_Original')
input_dir_path = os.path.join('Neural_Net', 'assets', 'Skin_Label_Data')
dest_dir_path = os.path.join('assets', 'Pulse_Data', '')
dest_skin_dir_path = os.path.join('assets', 'Skin_Label_Data', '')
file = '00143.MTS'
file_path = os.path.join(input_dir_path, file)


print(input_dir_path)

for file in os.listdir(input_dir_path):
    if file.endswith(".npy"):
        file_path = os.path.join(input_dir_path, file)

        print(file_path)

        # img = misc.imread(file_path)

        mean_img = np.load(file_path)
        #
        # mean_img = np.mean(img, axis=2)
        # white_index = mean_img >= 200.0
        #
        #
        # skin_index = mean_img < 200.0
        #
        # mean_img[skin_index] = 1
        # mean_img[white_index] = 0

        fig = plt.figure(figsize=(17, 9))
        sub1 = fig.add_subplot(111)
        sub1.set_title('Norm. Avg.')
        sub1.imshow(mean_img)
        # fig.savefig('assets\\Skin_00149.jpg')

        print(np.amin(mean_img))
        print(np.amax(mean_img))

        # np.save('Skin_00149.npy', mean_img)

        plt.show()



