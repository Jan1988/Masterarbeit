import numpy as np
from matplotlib import pyplot as plt
import time
import cv2
import os

from Video_Tools import load_video
from Video_Tools import split_vid_into_rgb_channels
from Video_Tools import get_video_dimensions
from Video_Tools import devide_frame_into_roi_means


def normalization(colour_channel_values):
    mean = np.mean(colour_channel_values)
    normalized_values = colour_channel_values - mean

    return normalized_values


# def overlap_add(input_signal, window_size=32, window_count=8):
#
#     L = len(input_signal)
#     overlap_signal = np.zeros((L), dtype='float64')
#
#     offsets = np.linspace(0, L - window_size, window_count)
#
#     hanning_window = np.hanning(window_size)
#
#     for n in offsets:
#         int_n = int(n)
#         hann_window_signal = hanning_window * input_signal[int_n:int_n + window_size]
#         overlap_signal[int_n/2:int_n/2 + window_size] += hann_window_signal
#
#     return overlap_signal

# Calculate Pulse Signal and BPM value for every ROI
def chrom_based_pulse_signal_estimation(fps, red_temp_array, green_temp_array, blue_temp_array):

    normalized_red_temp_array = normalization(red_temp_array)
    normalized_green_temp_array = normalization(green_temp_array)
    normalized_blue_temp_array = normalization(blue_temp_array)

    # Chrominance Signal X & Y
    chrom_x = 3 * normalized_red_temp_array - 2 * normalized_green_temp_array
    chrom_y = 1.5 * normalized_red_temp_array + normalized_green_temp_array - 1.5 * normalized_blue_temp_array

    # Standard deviation of x & y
    std_dev_x = np.std(chrom_x)
    std_dev_y = np.std(chrom_y)

    # alpha
    alpha = std_dev_x / std_dev_y

    # pulse signal S
    S = chrom_x - alpha * chrom_y

    # Hann window signal
    hann_window = np.hanning(len(S))
    hann_window_signal = hann_window * S

    # Fourier Transform
    raw = np.fft.fft(hann_window_signal, 512)
    L = int(len(raw) / 2 + 1)
    fft1 = np.abs(raw[:L])

    frequencies = np.linspace(0, fps / 2, L, endpoint=True)
    heart_rates = frequencies * 60

    # bandpass filter for pulse
    bound_low = (np.abs(heart_rates - 55)).argmin()
    bound_high = (np.abs(heart_rates - 180)).argmin()
    fft1[:bound_low] = 0
    fft1[bound_high:] = 0

    max_freq_pos = np.argmax(fft1)

    roi_bpm = heart_rates[max_freq_pos]

    return roi_bpm, heart_rates, fft1, hann_window_signal, S


if __name__ == '__main__':
    start_time = time.time()

    dir_path = os.path.join('assets', 'Vid_Original')
    file = '00100.MTS'
    file_path = os.path.join(dir_path, file)
    w_div = 16
    h_div = 8
    bpm_values = np.zeros((h_div, w_div), dtype='float64')

    print(file_path)
    vid_data, fps = load_video(file_path)
    vid_data = vid_data[1:250]

    frame_count, width, height = get_video_dimensions(vid_data)
    print('Cutted length: ' + str(frame_count))
    # w_steps = width/w_div
    # h_steps = height/h_div
    roi_mean_frames = np.zeros((frame_count, w_div, h_div, 3), dtype='float64')

    #
    for j, frame in enumerate(vid_data):

        #
        # Spatial Averaging
        roi_means_2DArray, frame_devided = devide_frame_into_roi_means(frame, w_div, h_div)

        #
        # Create time series array of the roi means
        roi_mean_frames[j] = roi_means_2DArray

    red_vid_frames, green_vid_frames, blue_vid_frames = split_vid_into_rgb_channels(roi_mean_frames)

    #
    # Für die Darstellung der Puls Ergebnismatrix
    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(file, fontsize=14, fontweight='bold')
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    #
    #

    '''BPM Estimation for every ROI'''
    for x in range(0, w_div):
        for y in range(0, h_div):
            bpm, heart_rates, fft, hann_S, S = chrom_based_pulse_signal_estimation(fps, red_vid_frames[:, x, y], green_vid_frames[:, x, y], blue_vid_frames[:, x, y])

            bpm_values[y, x] = bpm
            sub2.text(x, y, round(bpm, 1), color=(0.745, 0.467, 0.294), fontsize=8, va='center', ha='center')

    print(bpm_values)

    sub1.set_title('h-Signal')
    sub1.imshow(frame_devided)

    sub2.set_title('BPM Matrix')
    sub2.matshow(bpm_values, cmap=plt.cm.gray)

    # plt.matshow(bpm_values, cmap=plt.cm.gray)
    plt.tight_layout()
    # fig.savefig(file_path[:-4] + '.jpg')
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
