import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Video_Tools import load_video, get_video_dimensions
from CHROM_Based_Method import chrom_based_pulse_signal_estimation
from POS_Based_Method import pos_based_method
from Preprocessing_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals


def plot_results(fft, heart_rates, overlap_signal=0, raw=0, pulse_signal=0, green_norm=0):
    # plt.axis([0, n, y_lower, y_upper])

    fig = plt.figure(figsize=(17, 9))

    sub1 = fig.add_subplot(331)
    sub1.set_title('Norm. Avg.')
    # sub1.plot(int_frames, red_norm, 'r',
    #           int_frames, green_norm, 'g',
    #           int_frames, blue_norm, 'b')
    sub1.plot(green_norm, 'g')

    sub2 = fig.add_subplot(332)
    sub2.set_title('Pulse Signal')
    sub2.plot(pulse_signal, 'k')

    sub5 = fig.add_subplot(333)
    sub5.set_title('Overlap-added Signal')
    sub5.plot(overlap_signal, 'k')

    sub8 = fig.add_subplot(334)
    sub8.set_title('Hanning Window')
    sub8.plot(raw, 'k')

    sub7 = fig.add_subplot(335)
    sub7.set_title('FFT abs')
    sub7.plot(heart_rates, fft, 'k')

    plt.show()




if __name__ == '__main__':
    start_time = time.time()

    dir_path = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    file = '00130.MTS'

    file_path = os.path.join(dir_path, file)

    w_div = 16
    h_div = 8

    skin_mat = np.zeros((h_div, w_div), dtype='float64')

    pulse_label_data = load_label_data()


    for file in os.listdir(dir_path):
        if file.endswith(".MTS"):
            file_path = os.path.join(dir_path, file)
            print(file_path)
            video_frames, fps = load_video(file_path)
            video_frames = video_frames[1:250]
            frame_count, width, height = get_video_dimensions(video_frames)
            w_steps = int(width / w_div)
            h_steps = int(height / h_div)
            pulse_lower, pulse_upper = get_pulse_vals_from_label_data(pulse_label_data, file)

            # Für die Darstellung der Puls Ergebnismatrix
            fig = plt.figure(figsize=(17, 9))
            fig.suptitle(file, fontsize=14, fontweight='bold')
            sub1 = fig.add_subplot(121)
            sub2 = fig.add_subplot(122)
            last_frame_clone = video_frames[frame_count - 1].copy()

            '''BPM Estimation for every ROI'''
            for x in range(0, width, w_steps):
                for y in range(0, height, h_steps):
                    bpm, fft, heart_rates, raw, hann_w, S, green_avg = chrom_based_pulse_signal_estimation(video_frames[:, x:x+w_steps, y:y+h_steps], fps)

                    sub1.text(x+w_steps/2, y+h_steps/2, round(bpm, 1), color=(0.0, 0.0, 0.0), fontsize=7, va='center', ha='center')
                    cv2.rectangle(last_frame_clone, (x, y), (x + w_steps, y + h_steps), (0, 0, 0), 2)

                    skin_mat[int(y/h_steps), int(x/w_steps)] = compare_pulse_vals(bpm, pulse_lower, pulse_upper)

            sub1.set_title('BPM on ROIs')
            sub1.imshow(last_frame_clone)
            sub2.set_title('Skin, Non-Skin Matrix')
            sub2.matshow(skin_mat, cmap=plt.cm.gray)

            plt.tight_layout()
            fig.savefig(file_path[:-4] + '.jpg')
            # plt.show()

    # plot_results(fft, heart_rates, raw, hann_w, S, green_avg)
    # print(bpm_values)

    print("--- %s seconds ---" % (time.time() - start_time))
