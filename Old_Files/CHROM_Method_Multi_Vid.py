import os
import time

import numpy as np
from matplotlib import pyplot as plt

from Old_Files.CHROM_Method_Single_Vid import chrom_based_pulse_signal_estimation
from Video_Tools import devide_frame_into_roi_means
from Video_Tools import get_video_dimensions
from Video_Tools import load_video
from Video_Tools import split_vid_into_rgb_channels


def normalization(colour_channel_values):
    mean = np.mean(colour_channel_values)
    normalized_values = colour_channel_values - mean

    return normalized_values


if __name__ == '__main__':
    start_time = time.time()

    dir_path = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    w_div = 32
    h_div = 16
    bpm_values = np.zeros((h_div, w_div), dtype='float64')

    for file in os.listdir(dir_path):
        if file.endswith(".MTS"):
            file_path = os.path.join(dir_path, file)
            print(file_path)
            video_frames, fps = load_video(file_path)

            video_frames = video_frames[1:250]

            frame_count, width, height = get_video_dimensions(video_frames)
            print('Cutted length: ' + str(frame_count))
            # w_steps = width/w_div
            # h_steps = height/h_div
            roi_mean_frames = np.zeros((frame_count, w_div, h_div, 3), dtype='float64')

            for j, frame in enumerate(video_frames):

                '''Spatial Averaging
                The video frame is split into w_div * h_div ROIs
                For every ROI the mean of all ROI pixels is calculated
                Return is an 2D Array with the mean values of every ROI and
                the original image with rectangles displaying each ROI
                '''
                # blurred_frame = cv2.GaussianBlur(frame, (5, 5), 1)
                roi_means_2DArray, frame_devided = devide_frame_into_roi_means(frame, w_div, h_div)

                # Create time series array of the roi means
                roi_mean_frames[j] = roi_means_2DArray

            red_vid_frames, green_vid_frames, blue_vid_frames = split_vid_into_rgb_channels(roi_mean_frames)

            # Für die Darstellung der Puls Ergebnismatrix
            fig = plt.figure(figsize=(17, 9))
            fig.suptitle(file, fontsize=14, fontweight='bold')
            sub1 = fig.add_subplot(121)
            sub2 = fig.add_subplot(122)

            '''BPM Estimation for every ROI'''
            for x in range(0, w_div):
                for y in range(0, h_div):

                    bpm, heart_rates, fft, hann_S, S = chrom_based_pulse_signal_estimation(fps, red_vid_frames[:, x, y], green_vid_frames[:, x, y], blue_vid_frames[:, x, y])

                    bpm_values[y, x] = bpm
                    sub2.text(x, y, round(bpm, 1), color=(0.745, 0.467, 0.294), fontsize=8, va='center', ha='center')

            print(bpm_values)

            sub1.set_title('h-Signal')
            sub1.imshow(frame_devided)

            sub2.set_title('BPM Matrix')
            sub2.matshow(bpm_values, cmap=plt.cm.gray)

            # plt.matshow(bpm_values, cmap=plt.cm.gray)
            plt.tight_layout()
            fig.savefig(file_path[:-4] + '.jpg')
            # plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
