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

    # Color Channel Buffers
    red_vid_data_buffer = []
    green_vid_data_buffer = []
    blue_vid_data_buffer = []
    buffer_size = 128

    filename = 'assets\\output_2.mp4'
    vid_data, fps = load_video(filename)

    width, height = get_frame_dimensions(vid_data[0])


    i = 0
    for frame in vid_data:

        red_frame, green_frame, blue_frame = split_into_rgb_channels(frame)

        # Mean for every colour channel over their pixel values
        red_vid_data_buffer.append(np.mean(red_frame))
        green_vid_data_buffer.append(np.mean(green_frame))
        blue_vid_data_buffer.append(np.mean(blue_frame))

        L = len(red_vid_data_buffer)

        # Throw out first value of buffer to ensure correct length
        if L > buffer_size:
            red_vid_data_buffer.pop(0)
            green_vid_data_buffer.pop(0)
            blue_vid_data_buffer.pop(0)
            L = buffer_size

        # When buffer is filled start calculations
        if L > 127:
            # Normalization
            normalized_roi_red_values = normalization(red_vid_data_buffer)
            normalized_roi_green_values = normalization(green_vid_data_buffer)
            normalized_roi_blue_values = normalization(blue_vid_data_buffer)

            # bandpassed_red = temporal_bandpass_filter(normalized_roi_red_values, 30)
            # bandpassed_green = temporal_bandpass_filter(normalized_roi_green_values, 30)
            # bandpassed_blue = temporal_bandpass_filter(normalized_roi_blue_values, 30)

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
            # s2 = 3 * (1 - alpha / 2) * bandpassed_red - 2 * (1 + alpha / 2) * bandpassed_green + 3 * alpha / 2 * bandpassed_blue

            plt.subplot(223)
            plt.xlabel("Frames")
            plt.ylabel("Pixel Average")
            plt.plot(s)
            plt.title('mean_values Image')
            plt.xticks([])
            plt.yticks([])

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        i += 1
        print('Frame: ' + str(i))


    print("--- %s seconds ---" % (time.time() - start_time))