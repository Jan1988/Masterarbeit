
import os
import time
import numpy as np

from matplotlib import pyplot as plt

from Video_Tools import load_video, get_video_dimensions
from POS_Based_Method import pos_based_method, extract_pos_based_method_improved
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin, \
    save_rois_with_label


def extr_roi_multi_video_calculation(in_dir, out_dir):

    for file in os.listdir(in_dir):
        in_file_path = os.path.join(in_dir, file)

        if file.endswith(".MTS"):
            extr_roi_single_video_calculation(file, in_file_path, out_dir)



# Old
def extr_pos_based_method(time_series, fps):

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


# For ROI
def extr_roi_single_video_calculation(in_file, in_file_path, out_dir):

    w_div = 64
    h_div = 32

    video_frames, fps = load_video(in_file_path)
    video_frames = video_frames[22:310]
    frame_count, width, height = get_video_dimensions(video_frames)
    w_steps = int(width / w_div)
    h_steps = int(height / h_div)

    # Riesen-ndarray für puls-signale für breite*höhe eines Videos
    pulse_signal_data = np.zeros([h_div, w_div, 44], dtype='float64')


    # Hier wird der ungeradere Rest abgeschnitten
    width = w_steps * w_div
    height = h_steps * h_div
    for x in range(0, width, w_steps):
        for y in range(0, height, h_steps):
            roi_ind_x = int(x / w_steps)
            roi_ind_y = int(y / h_steps)

            roi_time_series = video_frames[:, y:y+h_steps, x:x+w_steps]
            # Spatial Averaging
            roi_time_series_avg = np.mean(roi_time_series, axis=(1, 2))

            puls_signal = extract_pos_based_method_improved(roi_time_series_avg, fps)

            pulse_signal_data[roi_ind_y, roi_ind_x] = puls_signal

        print("Fortschritt: %.1f %%" % ((x+1) / width*100))

    print("--- Extr Finished %s seconds ---" % (time.time() - start_time))
    # print(time.perf_counter())

    out_file_path = os.path.join(out_dir, 'ROI_' + in_file[:-4] + '.npy')
    np.save(out_file_path, pulse_signal_data)


# For every pixel
def extr_single_video_calculation(file, file_path):

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[22:310]
    frame_count, width, height = get_video_dimensions(video_frames)

    # Riesen-ndarray für puls-signale für breite*höhe eines Videos
    _pulse_signal_data = np.zeros([height, width, 44], dtype='float64')

    for x in range(0, width):
        for y in range(0, height):

            px_time_series = video_frames[:, y, x]

            puls_signal = extract_pos_based_method_improved(px_time_series, fps)

            _pulse_signal_data[y, x] = puls_signal

        # print('Bildpunkt: ' + str(x))
        print("Fortschritt: %.1f %%" % ((x+1) / width*100))

    # reshape to fit in .txt file
    # reshaped_pulse_signal_data = _pulse_signal_data.reshape(height * width, _pulse_signal_data.shape[2])

    print("--- Extr Finished %s seconds ---" % (time.time() - start_time))
    # print(time.perf_counter())

    return _pulse_signal_data


if __name__ == '__main__':

    start_time = time.time()
    file = '00163.MTS'
    Pulse_data_dir = os.path.join('assets', 'Pulse_Data')
    video_dir = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    video_file_path = os.path.join(video_dir, file)

    # pulse_signal_data = extr_single_video_calculation(file, file_path)
    # extr_roi_single_video_calculation(file, video_file_path, Pulse_data_dir)
    extr_roi_multi_video_calculation(video_dir, Pulse_data_dir)

    print("--- Algorithm Completed %s seconds ---" % (time.time() - start_time))
