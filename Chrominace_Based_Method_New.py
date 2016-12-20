import numpy as np
from matplotlib import pyplot as plt
import time
import cv2
import scipy
import os

from Video_Tools import load_video
from Video_Tools import temporal_bandpass_filter
from Video_Tools import split_vid_into_rgb_channels
from Video_Tools import get_frames_dimension
from Video_Tools import get_frame_dimensions


def devide_image_into_roi_means(image, div_width, div_height):

    width, height = get_frame_dimensions(image)
    roi_width = int(width / div_width)
    roi_height = int(height / div_height)

    roi_means_2darray = np.zeros((div_height, div_width, 3), dtype='float64')

    # Ungeradere Rest wird abgeschnitten
    width = roi_width * div_width
    height = roi_height * div_height
    for x in range(0, width, roi_width):
        for y in range(0, height, roi_height):

            cv2.rectangle(image, (x, y), (x + roi_width, y + roi_height), (0, 0, 0), 1)
            roi = image[y:y + roi_height, x:x + roi_width]

            #2D array wird mit den Means der ROIs gefüllt
            roi_means_2darray[(x/roi_width), (y/roi_height)] = np.mean(roi, axis=(0, 1))

    return roi_means_2darray, image


def normalization(colour_channel_values):
    mean = np.mean(colour_channel_values)
    normalized_values = colour_channel_values - mean

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

    file_path = os.path.join('assets', '00101.mp4')
    # file_path = os.path.join('assets', 'ROIs', '00101.mp4')
    vid_data, fps = load_video(file_path)

    L, width, height = get_frames_dimension(vid_data)
    print('Frames: ' + str(L))

    roi_mean_frames = np.zeros((L, 16, 16, 3), dtype='float64')
    bpm_values = np.zeros((16, 16), dtype='float64')

    j = 0
    for frame in vid_data:

        # Spatial Averaging
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 1)
        roi_means_2DArray, frame_devided = devide_image_into_roi_means(blurred_frame, 16, 16)

        # Create time series array of the roi means
        roi_mean_frames[j] = roi_means_2DArray

        # Show results
        cv2.imshow('Frames', frame_devided)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        j += 1

    # plt.matshow(roi_means_2DArray[:, :, 0], cmap=plt.cm.gray)
    # plt.show()

    red_vid_frames, green_vid_frames, blue_vid_frames = split_vid_into_rgb_channels(roi_mean_frames)

    # Normalization
    for y in range(0, roi_mean_frames.shape[1]):
        for x in range(0, roi_mean_frames.shape[2]):

            red_temp_array = red_vid_frames[:, y, x]
            green_temp_array = green_vid_frames[:, y, x]
            blue_temp_array = blue_vid_frames[:, y, x]

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
            s = chrom_x - alpha * chrom_y

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

            bpm_values[y, x] = bpm

    plt.matshow(bpm_values, cmap=plt.cm.gray)
    plt.show()
    print('average bpm: ' + str(np.mean(bpm_values)))


    # plt.subplot(223)
    # plt.xlabel("Frames")
    # plt.ylabel("Pixel Average")
    # plt.plot(s)
    # plt.title('Signal Values Image')
    # plt.xticks([])
    # plt.yticks([])
    #
    # # Display images
    # plt.subplot(221)
    # plt.imshow(frame)
    # plt.title('frame')
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(222)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.plot(fft1)
    # plt.title('fft1 Image')
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(224)
    # plt.xlabel("Frames")
    # plt.ylabel("Pixel Average")
    # plt.plot(pruned)
    # plt.title('pruned')
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
