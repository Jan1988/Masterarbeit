
import os
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt

from Video_Tools import load_video, get_video_dimensions, normalize_mean_over_interval
from CHROM_Based_Method import chrom_based_pulse_signal_estimation
from POS_Based_Method import pos_based_method
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin, \
    save_rois_with_label


def extr_pos_based_method(time_series, fps):
    # sliding window size
    window_size = 48

    sequence_length = len(time_series)
    H = np.zeros(sequence_length, dtype='float64')
    red = time_series[:, 2]
    blue = time_series[:, 1]
    green = time_series[:, 0]

    red_buf = []
    blue_buf = []
    green_buf = []

    for n in range(1, sequence_length):

        red_buf.append(red[n])
        blue_buf.append(blue[n])
        green_buf.append(green[n])

        m = n - window_size + 1
        if m > 0:
            # print('Window from ' + str(m-1) + ' ' + str(n) + ' length ' + str(len(H[m-1:n])))

            # 5 temporal normalization
            red_norm = normalize_mean_over_interval(red_buf[m - 1:n])
            green_norm = normalize_mean_over_interval(green_buf[m - 1:n])
            blue_norm = normalize_mean_over_interval(blue_buf[m - 1:n])

            # 6 projection
            S1 = 0 * red_norm + 1 * green_norm - 1 * blue_norm
            S2 = -2 * red_norm + 1 * green_norm + 1 * blue_norm

            # 7 tuning
            S1_std = np.std(S1)
            S2_std = np.std(S2)

            alpha = S1_std / S2_std

            h = S1 + alpha * S2

            # make h zero-mean
            h_mean = np.mean(h)
            h_no_mean = h - h_mean

            H[m - 1:n] += h_no_mean

    # Fourier Transform
    raw = np.fft.fft(H, 512)
    L = int(len(raw) / 2 + 1)
    fft1 = np.abs(raw[:L])

    # fig = plt.figure(figsize=(17, 9))
    #
    # sub1 = fig.add_subplot(221)
    # sub1.set_title('H')
    # sub1.plot(H, 'g')
    #
    # sub2 = fig.add_subplot(222)
    # sub2.set_title('fft1')
    # sub2.plot(fft1, 'k')

    frequencies = np.linspace(0, fps / 2, L, endpoint=True)
    heart_rates = frequencies * 60

    # bandpass filter for pulse
    bound_low = (np.abs(heart_rates - 50)).argmin()
    bound_high = (np.abs(heart_rates - 180)).argmin()
    fft1[:bound_low] = 0
    fft1[bound_high:] = 0

    # sub4 = fig.add_subplot(224)
    # sub4.set_title('fft1')
    # sub4.plot(fft1, 'k')
    #
    # plt.show()

    return fft1[bound_low:bound_high]


def extr_single_video_calculation(file, file_path):

    w_div = 16
    h_div = 8

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[50:350]
    frame_count, width, height = get_video_dimensions(video_frames)

    # Riesen-ndarray für puls-signale für breite*höhe eines Videos
    _pulse_signal_data = np.zeros([width, height, 44], dtype='float64')

    for x in range(0, width):
        for y in range(0, height):

            roi_time_series = video_frames[:, y, x]
            print("1--- %s seconds ---" % (time.time() - start_time))

            puls_signal = extr_pos_based_method(roi_time_series, fps)
            print("2--- %s seconds ---" % (time.time() - start_time))

            _pulse_signal_data[x, y, :] = puls_signal
            print("3--- %s seconds ---" % (time.time() - start_time))
        print('Fortschritt: ' + str(int(x / width)) + '%')

    return _pulse_signal_data


if __name__ == '__main__':

    start_time = time.time()
    # dir_path = os.path.join('assets', 'Vid_Original')
    dir_path = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    file = '00130.MTS'
    file_path = os.path.join(dir_path, file)



    # pulse_label_data = load_label_data()

    pulse_signal_data = extr_single_video_calculation(file, file_path)

    dest_folder = os.path.join('assets', 'Pulse_Data', file[:-4], '')
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    file_path_out = dest_folder + file + '.txt'
    with open(dest_folder, 'wb') as outfile:
        np.savetxt(outfile, pulse_signal_data, fmt='%i')

    print("--- %s seconds ---" % (time.time() - start_time))