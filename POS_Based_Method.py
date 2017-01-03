
import numpy as np

from Video_Tools import split_frame_into_rgb_channels, normalize_mean_over_interval


# def overlap_add(traces, window_len, overlap):
#     window_spacing = window_len - overlap
#     num_windows = int((len(traces)-overlap) / window_spacing)


def pos_based_method(video_frames, fps):

    sequence_length = len(video_frames)

    # sliding window size
    window_size = 48

    #
    H = np.zeros(sequence_length, dtype='float64')
    # Color Channel Buffers
    red_list = []
    green_list = []
    blue_list = []

    n = 1
    for frame in video_frames:

        red_frame, green_frame, blue_frame = split_frame_into_rgb_channels(frame)

        #3 Spatial Averaging
        red_frame_avg = np.mean(red_frame)
        green_frame_avg = np.mean(green_frame)
        blue_frame_avg = np.mean(blue_frame)

        red_list.append(red_frame_avg)
        green_list.append(green_frame_avg)
        blue_list.append(blue_frame_avg)

        m = n - window_size + 1
        if m > 0:
            # print('Window from ' + str(m-1) + ' ' + str(n) + ' length ' + str(len(H[m-1:n])))

            # 5 temporal normalization
            red_norm = normalize_mean_over_interval(red_list[m-1:n])
            green_norm = normalize_mean_over_interval(green_list[m-1:n])
            blue_norm = normalize_mean_over_interval(blue_list[m-1:n])

            # 6 projection
            S1 = 0 * red_norm + 1 * green_norm - 1 * blue_norm
            S2 = -2 * red_norm + 1 * green_norm + 1 * blue_norm

            # 7 tuning
            S1_std = np.std(S1)
            S2_std = np.std(S2)

            alpha = S1_std / S2_std

            h = S1 + alpha * S2

            #make h zero-mean
            h_mean = np.mean(h)
            h_no_mean = h - h_mean

            H[m-1:n] += h_no_mean

        n += 1

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

    return bpm, fft1, heart_rates, raw, H, h_no_mean, green_list



