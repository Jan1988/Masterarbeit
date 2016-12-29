import time
import os
import numpy as np
import scipy


from matplotlib import pyplot as plt
from Video_Tools import load_video
from Video_Tools import get_video_dimensions
from Video_Tools import split_frame_into_rgb_channels
from Video_Tools import temporal_bandpass_filter


# temporal deviding through mean normalization
def normalize(list):

    list_mean = np.mean(list)
    normalized_list = list/ list_mean

    return normalized_list


if __name__ == '__main__':
    start_time = time.time()

    # sliding window size
    window_size = 32

    # Color Channel Buffers
    red_list = []
    green_list = []
    blue_list = []

    file_path = os.path.join('assets', 'ROIs', 'new_00100.mp4')

    video_frames, fps = load_video(file_path)
    print('fps: ' + str(fps))
    cutted_frames = video_frames[2:]
    frame_count, width, height = get_video_dimensions(cutted_frames)

    H = np.zeros((1, frame_count), dtype='float64')

    h = np.zeros((1, frame_count), dtype='float64')
    n = 1
    for frame in cutted_frames:

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
            # 5 temporal normalization
            red_norm = normalize(red_list)
            green_norm = normalize(green_list)
            blue_norm = normalize(blue_list)

            # 6 Projection
            S1 = 0 * red_norm + 1 * green_norm - 1 * blue_norm
            S2 = -2 * red_norm + 1 * green_norm + 1 * blue_norm

            S1_std = np.std(S1)
            S2_std = np.std(S2)

            alpha = S1_std / S2_std

            h = S1 + alpha * S2

            #make h zero-mean

            h_mean = np.mean(h)
            h_no_mean = h - h_mean

        n += 1



    int_frames = list(range(1, n))
    signal_frames = list(range(1, n))
    # 1. X-Achse 2. Y-Achse
    # plt.axis([0, n, y_lower, y_upper])

    fig = plt.figure(figsize=(17, 9))

    sub1 = fig.add_subplot(331)
    sub1.set_title('Norm. Avg.')
    # sub1.plot(int_frames, red_norm, 'r',
    #           int_frames, green_norm, 'g',
    #           int_frames, blue_norm, 'b')
    sub1.plot(int_frames, green_norm, 'g')

    sub2 = fig.add_subplot(332)
    sub2.set_title('S1 & S2 Signals')
    sub2.plot(int_frames, S1, 'm',
              int_frames, S2, 'c')

    sub3 = fig.add_subplot(333)
    sub3.set_title('h-Signal')
    sub3.plot(int_frames, h, 'k')

    sub4 = fig.add_subplot(334)
    sub4.set_title('h-zero-mean')
    sub4.plot(int_frames, h_no_mean, 'k')

    # Fourier Transform
    hann_window_signal = np.hanning(frame_count) * h_no_mean

    raw = np.fft.fft(hann_window_signal, 512)
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
    print(bpm)


    # # FFT 2
    # freqs2 = scipy.fftpack.fftfreq(len(h_no_mean), d=1.0 / fps)
    # fft2 = abs(scipy.fftpack.fft(h_no_mean))
    #
    # idx = np.argsort(freqs2)
    # freqs2 = freqs2[idx]
    # fft2 = fft2[idx]
    # freqs2 = freqs2[len(freqs2) / 2 + 1:] * 60.0
    # fft2 = fft2[len(fft2) / 2 + 1:]

    sub5 = fig.add_subplot(335)
    sub5.set_title('Hanning Window')
    sub5.plot(hann_window_signal, 'k')

    sub6 = fig.add_subplot(336)
    sub6.set_title('FFT Raw')
    sub6.plot(raw, 'k')

    sub7 = fig.add_subplot(337)
    sub7.set_title('FFT abs')
    sub7.plot(heart_rates, fft1, 'k')

    # sub7 = fig.add_subplot(337)
    # sub7.set_title('FFT fft2')
    # sub7.plot(freqs2, fft2, 'k')

    # sub8 = fig.add_subplot(338)
    # sub8.set_title('FFT abs')
    # sub8.plot(fft1, 'k')

    plt.show()




