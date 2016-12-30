import numpy as np
import os
import cv2

from Video_Tools import devide_frame_into_roi_means
from Video_Tools import load_video
from matplotlib import pyplot as plt


dir_path = os.path.join('assets', 'Vid_Original')
file = '00100.MTS'
file_path = os.path.join(dir_path, file)

video_frames, fps = load_video(file_path)
video_frames = video_frames[1:100]

buffer_size = 32
buffer_size_half = int(buffer_size/2)
frame_buffer = []

j = 0
for frame in video_frames:

    buffer_count = len(frame_buffer)

    if buffer_count >= buffer_size:
        frame_buffer = frame_buffer[-buffer_size_half:]

    frame_buffer.append(j)
    print(frame_buffer)
    j += 1


