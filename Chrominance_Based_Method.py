import numpy as np
from matplotlib import pyplot as plt
import time
import cv2
import scipy

from Video_Tools import load_video
from Video_Tools import temporal_bandpass_filter
from Video_Tools import split_into_rgb_channels
from Video_Tools import get_frame_dimensions


def normalization(colour_channel_values):
    mean = np.mean(colour_channel_values)
    normalized_values = colour_channel_values / mean

    return normalized_values


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


if __name__ == '__main__':
    start_time = time.time()

    filename = 'assets\\output_2.mp4'
    vid_data, fps = load_video(filename)

    # Color Channel Arrays
    center_values = []
    mean_values = []

    point_red_values = []
    point_green_values = []
    point_blue_values = []

    roi_red_values = []
    roi_green_values = []
    roi_blue_values = []

    width, height = get_frame_dimensions(vid_data[0])
    center = (int(width/2), int(height/2))
    center_add = (int(width/2)+20, int(height/2)+20)

    roi = vid_data[:, int(width/2):int(width/2)+20, int(height/2):int(height/2)+20, :]



    i = 0
    for frame in vid_data:

        roi_frame = roi[i]
        center_values.append(frame[center])

        red_frame, green_frame, blue_frame = split_into_rgb_channels(frame)

        point_red_values.append(red_frame[center])
        point_green_values.append(green_frame[center])
        point_blue_values.append(blue_frame[center])

        roi_red_values.append(np.mean(red_frame))
        roi_green_values.append(np.mean(green_frame))
        roi_blue_values.append(np.mean(blue_frame))

        # cv2.imshow('red_frame', red_frame)
        # cv2.imshow('green_frame', green_frame)
        # cv2.imshow('blue_frame', blue_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    # Normalization
    normalized_roi_red_values = normalization(roi_red_values)
    normalized_roi_green_values = normalization(roi_green_values)
    normalized_roi_blue_values = normalization(roi_blue_values)

    bandpassed_red = temporal_bandpass_filter(normalized_roi_red_values, 30)
    bandpassed_green = temporal_bandpass_filter(normalized_roi_green_values, 30)
    bandpassed_blue = temporal_bandpass_filter(normalized_roi_blue_values, 30)


    # Chrominance Signal x & y
    x = 3 * normalized_roi_red_values - 2 * normalized_roi_green_values
    y = 1.5 * normalized_roi_red_values + normalized_roi_green_values - 1.5 * normalized_roi_blue_values

    # Standard deviation of x & y
    std_dev_x = np.std(x)
    std_dev_y = np.std(y)

    # alpha
    alpha = std_dev_x / std_dev_y

    # pulse signal S
    s = x - alpha * y

    # s = temporal_bandpass_filter(s, 30)

    s2 = 3 * (1-alpha/2) * bandpassed_red - 2 * (1+alpha/2) * bandpassed_green + 3 * alpha / 2 * bandpassed_blue

    plt.subplot(223)
    plt.xlabel("Frames")
    plt.ylabel("Pixel Average")
    plt.plot(s)
    plt.title('mean_values Image')
    plt.xticks([])
    plt.yticks([])

    L = len(s)

    hanning_window = np.hanning(L)

    s = hanning_window * s

    raw = np.fft.rfft(s, 512)

    fft1 = np.abs(raw)

    arranged = np.arange(L / 2 + 1)

    freqs = fps / L * arranged

    freqs_new = 60. * freqs

    idx = np.where((freqs_new > 50) & (freqs_new < 210))

    pruned = fft1[idx]
    freqs_pruned = freqs_new[idx]
    idx2 = np.argmax(pruned)
    bpm = freqs_pruned[idx2]
    print(bpm)

    # Fourier Transformation
    # freqs = scipy.fftpack.fftfreq(len(s), d=1.0 / fps)
    # fft = abs(scipy.fftpack.fft(s))
    #
    # idx = np.argsort(freqs)
    # freqs = freqs[idx]
    # fft = fft[idx]
    #
    # freqs = freqs[len(freqs) / 2 + 1:]
    # fft = fft[len(fft) / 2 + 1:]



    i -= 1
    # Display images
    plt.subplot(221)
    plt.imshow(frame)
    plt.title('frame')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.plot(fft1)
    plt.title('fft1 Image')
    plt.xticks([])
    plt.yticks([])

    # plt.subplot(223)
    # plt.xlabel("Frames")
    # plt.ylabel("Pixel Average")
    # plt.plot(s)
    # plt.title('mean_values Image')
    # plt.xticks([])
    # plt.yticks([])

    plt.subplot(224)
    plt.xlabel("Frames")
    plt.ylabel("Pixel Average")
    plt.plot(pruned)
    plt.title('pruned')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
