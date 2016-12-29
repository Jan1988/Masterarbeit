import numpy as np
from matplotlib import pyplot as plt
import time
import scipy
import os

from Video_Tools import load_video
from Video_Tools import load_video_float

start_time = time.time()

file_path = os.path.join('assets', 'ROIs', 'new_00101.mp4')
vid_data, fps = load_video_float(file_path)
"""Graph the average value of the video as well as the frequency strength"""
averages = []
bounds = None

cutted_frames = vid_data[2:]

if bounds:
    for x in range(1, cutted_frames.shape[0] - 1):
        averages.append(cutted_frames[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
else:
    for x in range(1, cutted_frames.shape[0] - 1):
        averages.append(cutted_frames[x, :, :, :].sum())

averages = averages - min(averages)

charts_x = 1
charts_y = 2
plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=.7)

plt.subplot(charts_y, charts_x, 1)
plt.title("Pixel Average")
plt.xlabel("Time")
plt.ylabel("Brightness")
plt.plot(averages)

print(len(averages))
freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
fft = abs(scipy.fftpack.fft(averages))

idx = np.argsort(freqs)

plt.subplot(charts_y, charts_x, 2)
plt.title("FFT")
plt.xlabel("Freq (Hz)")
freqs = freqs[idx]
fft = fft[idx]

freqs = freqs[len(freqs) / 2 + 1:]
fft = fft[len(fft) / 2 + 1:]
plt.plot(freqs, abs(fft))

plt.show()

print("--- %s seconds ---" % (time.time() - start_time))