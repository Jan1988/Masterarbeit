import os
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin,\
    save_rois_with_label
from Video_Tools import load_video, get_video_dimensions, normalize_mean_over_interval, split_frame_into_rgb_channels


def pos_based_method_improved(roi_time_series, fps):

    time_series = np.average(roi_time_series, axis=(1, 2))

    # func_time = time.time()

    # sliding window size
    window_size = 48

    sequence_length = len(time_series)
    H = np.zeros(sequence_length, dtype='float64')

    interval_count = int(sequence_length / window_size) * 2

    H = np.zeros(sequence_length, dtype='float64')

    n = window_size
    for i in range(interval_count):
        m = n - window_size

        # 5 temporal normalization
        window = time_series[m:n]
        mean_array = np.average(window, axis=0)
        norm_array = window / mean_array

        # 6 projection
        S1 = np.dot(norm_array, [-1, 1, 0])
        S2 = np.dot(norm_array, [1, 1, -2])

        # 7 tuning
        S1_std = np.std(S1)
        S2_std = np.std(S2)

        alpha = S1_std / S2_std

        h = S1 + alpha * S2

        # Hann window signal
        hann_window = np.hanning(len(h))
        hann_windowed_signal = hann_window * h

        # Overlap-adding
        H[m:n] += hann_windowed_signal
        n += int(window_size / 2)


    # Fourier Transform
    raw = np.fft.fft(H, 512)
    L = int(len(raw) / 2 + 1)
    fft1 = np.abs(raw[:L])

    frequencies = np.linspace(0, fps / 2, L, endpoint=True)
    heart_rates = frequencies * 60

    # bandpass filter for pulse
    bound_low = (np.abs(heart_rates - 50)).argmin()
    bound_high = (np.abs(heart_rates - 180)).argmin()
    fft1[:bound_low] = 0
    fft1[bound_high:] = 0

    max_freq_pos = np.argmax(fft1)

    bpm = heart_rates[max_freq_pos]

    # print("4--- %s seconds ---" % (time.time() - func_time))
    # print(time.perf_counter())
    # return fft1[bound_low:bound_high]
    return bpm, fft1, heart_rates, raw, H



if __name__ == '__main__':

    start_time = time.time()
    # dir_path = os.path.join('assets', 'Vid_Original')
    dir_path = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    file = '00130.MTS'
    file_path = os.path.join(dir_path, file)

    pulse_label_data = load_label_data()

    w_div = 16
    h_div = 8

    skin_mat = np.zeros((h_div, w_div), dtype='float64')
    bpm_values = np.zeros((h_div, w_div), dtype='float64')

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[50:350]
    frame_count, width, height = get_video_dimensions(video_frames)
    w_steps = int(width / w_div)
    h_steps = int(height / h_div)

    # Load all pulse value belonging to a certain video in array
    # Will be used for ROI labeling
    pulse_lower, pulse_upper = get_pulse_vals_from_label_data(pulse_label_data, file)

    # Für die Darstellung der Puls Ergebnismatrix
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(file, fontsize=14, fontweight='bold')
    sub1 = fig.add_subplot(221)
    sub2 = fig.add_subplot(222)
    sub3 = fig.add_subplot(223)
    sub4 = fig.add_subplot(224)

    last_frame = video_frames[frame_count - 1]
    last_frame_clone = last_frame.copy()

    '''BPM Estimation for every ROI'''
    for x in range(0, width, w_steps):
        for y in range(0, height, h_steps):
            roi_time_series = video_frames[:, y:y + h_steps, x:x + w_steps]

            bpm, fft1, heart_rates, raw, H = pos_based_method_improved(roi_time_series, fps)
            # h2 = pos_based_method(roi_time_series, fps)

            sub1.text(x+w_steps/2, y+h_steps/2, round(bpm, 1), color=(0.0, 0.0, 0.0), fontsize=7, va='center', ha='center')
            sub2.text(int(x/w_steps), int(y/h_steps), round(bpm, 1), color=(0.745, 0.467, 0.294), fontsize=8, va='center', ha='center')
            cv2.rectangle(last_frame_clone, (x, y), (x + w_steps, y + h_steps), (0, 0, 0), 2)

            bpm_values[int(y/h_steps), int(x/w_steps)] = bpm
            skin_mat[int(y/h_steps), int(x/w_steps)] = compare_pulse_vals(bpm, pulse_lower, pulse_upper)

    # check neighbouring rois
    bool_skin_mat = eliminate_weak_skin(skin_mat)
    save_rois_with_label(bool_skin_mat, last_frame, height, width, h_steps, w_steps, file[:-4])

    sub1.set_title('BPM on ROIs')
    sub1.imshow(last_frame_clone)
    sub2.set_title('BPM Matrix')
    sub2.matshow(bpm_values, cmap=plt.cm.gray)
    sub3.set_title('Skin, Non-Skin Matrix')
    sub3.matshow(skin_mat, cmap=plt.cm.gray)
    sub4.set_title('Skin, Neighbour reduced Matrix')
    sub4.matshow(bool_skin_mat, cmap=plt.cm.gray)

    plt.tight_layout()
    plt.show()
    print(time.perf_counter())












