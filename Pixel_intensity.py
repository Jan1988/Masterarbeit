
import time
import numpy as np
import cv2

from matplotlib import pyplot as plt
from Video_Tools import load_video
from Video_Tools import get_frames_dimension

start_time = time.time()

dir = 'assets\\Videos Original'
file = '00101.mts'

path = dir + '\\' + file
print(path)

vid_data, fps = load_video(path)

L, width, height = get_frames_dimension(vid_data)
print('Frames: ' + str(L))

intensity_frames = np.zeros((L, height, width, 1), dtype='float64')
intensity_array = []

i = 1
for frame in vid_data:

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_nxt = cv2.cvtColor(vid_data[i], cv2.COLOR_BGR2GRAY)

    diff = gray_frame - gray_frame_nxt
    intensity_between_frames = np.sum(np.abs(diff)) / (width * height)
    intensity_array.append(intensity_between_frames)
    i += 1
    if i == L:
        break


print("--- %s seconds ---" % (time.time() - start_time))
plt.xlabel("Frames")
plt.ylabel("Pixel Average")
plt.plot(intensity_array)
plt.title('Intensity Values')
plt.xticks([])
plt.yticks([])
plt.show()
