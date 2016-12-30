import numpy as np
import os
import cv2

from Video_Tools import devide_frame_into_roi_means
from Video_Tools import load_video
from matplotlib import pyplot as plt



file_path = os.path.join('assets', 'ROIs', 'new_00100.mp4')

video_frames, fps = load_video(file_path)

for frame in video_frames:

    # Spatial Averaging
    roi_means_2DArray, frame_devided = devide_frame_into_roi_means(frame, 8, 4)

    cv2.imshow('frame_devided', frame_devided)

print(roi_means_2DArray)
plt.imshow(frame_devided)
plt.show()
