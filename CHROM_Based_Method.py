import numpy as np

from Video_Tools import split_frame_into_rgb_channels, normalize_mean_over_interval


def chrom_based_pulse_signal_estimation(video_frames, fps):
    sequence_length = len(video_frames)

    # sliding window size
    # window_size = 48

    # Color Channel Buffers
    red_list = []
    green_list = []
    blue_list = []

    for frame in video_frames:

        red_frame, green_frame, blue_frame = split_frame_into_rgb_channels(frame)

        #3 Spatial Averaging
        red_frame_avg = np.mean(red_frame)
        green_frame_avg = np.mean(green_frame)
        blue_frame_avg = np.mean(blue_frame)

        red_list.append(red_frame_avg)
        green_list.append(green_frame_avg)
        blue_list.append(blue_frame_avg)

        normalized_red_temp_array = normalize_mean_over_interval(red_list)
        normalized_green_temp_array = normalize_mean_over_interval(green_list)
        normalized_blue_temp_array = normalize_mean_over_interval(blue_list)

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

    return roi_bpm, fft1, heart_rates, raw, hann_window_signal, S, green_list