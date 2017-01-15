import os
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin,\
    save_rois_with_label
from Video_Tools import load_video, get_video_dimensions, normalize_mean_over_interval, split_frame_into_rgb_channels


array = np.arange(60).reshape(4, 5, 3)


print(array[:, :, 0] > 20)



