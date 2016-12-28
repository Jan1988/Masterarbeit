import time
import os
import numpy as np


from matplotlib import pyplot as plt
from Video_Tools import load_video
from Video_Tools import get_video_dimensions
from Video_Tools import split_frame_into_rgb_channels


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

    file_path = os.path.join('assets', 'ROIs', 'new_00101.mp4')

    video_frames, fps = load_video(file_path)
    print('fps: ' + str(fps))
    cutted_frames = video_frames[1:]
    L, width, height = get_video_dimensions(cutted_frames)

    H = np.zeros((1, L), dtype='float64')

    h = np.zeros((1, L), dtype='float64')
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

    sub1 = fig.add_subplot(221)
    sub1.set_title('Norm. Avg.')
    # sub1.plot(int_frames, red_norm, 'r',
    #           int_frames, green_norm, 'g',
    #           int_frames, blue_norm, 'b')

    sub1.plot(int_frames, green_norm, 'r')

    sub2 = fig.add_subplot(222)
    sub2.set_title('S1 & S2 Signals')
    sub2.plot(int_frames, S1, 'm',
              int_frames, S2, 'c')

    sub3 = fig.add_subplot(223)
    sub3.set_title('h-Signal')
    sub3.plot(int_frames, h, 'k')

    sub4 = fig.add_subplot(224)
    sub4.set_title('h-zero-mean')
    sub4.plot(int_frames, h_no_mean, 'k')

    plt.show()


