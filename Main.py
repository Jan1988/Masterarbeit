import os
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt

from Video_Tools import load_video, get_video_dimensions
from CHROM_Based_Method import chrom_based_pulse_signal_estimation
from POS_Based_Method import pos_based_method, pos_based_method_improved
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin,\
    save_rois_with_label
from Skin_Detection import skin_detection_algorithm_multi_video


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

# def write_signals_to_txt(out_txt, signals_2darr):
#     with open(out_txt, 'wb') as outfile:
#         np.savetxt(outfile, signals_2darr, fmt='%i')


def multi_video_calculation(dir_path, pulse_label_data):

    for file in os.listdir(dir_path):
        if file.endswith(".MTS"):
            file_path = os.path.join(dir_path, file)
            print(file_path)
            single_video_calculation(file, file_path, pulse_label_data)


def single_video_calculation(file, file_path, pulse_label_data):
    w_div = 16
    h_div = 8

    skin_mat = np.zeros((h_div, w_div), dtype='float64')
    bpm_values = np.zeros((h_div, w_div), dtype='float64')

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[22:310]
    frame_count, width, height = get_video_dimensions(video_frames)
    w_steps = int(width / w_div)
    h_steps = int(height / h_div)

    # Load all pulse value belonging to a certain video in array
    # Will be used for ROI labeling
    pulse_lower, pulse_upper = get_pulse_vals_from_label_data(pulse_label_data, file)

    # Für die Darstellung der Puls Ergebnismatrix
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(file, fontsize=14, fontweight='bold')
    sub1 = fig.add_subplot(221)
    sub2 = fig.add_subplot(222)
    sub3 = fig.add_subplot(223)
    sub4 = fig.add_subplot(224)

    last_frame = video_frames[frame_count - 1]
    last_frame_clone = last_frame.copy()

    '''BPM Estimation for every ROI'''
    for x in range(0, width, w_steps):
        for y in range(0, height, h_steps):
            roi_time_series = video_frames[:, y:y+h_steps, x:x+w_steps]

            # Puls-Signal Extraction
            bpm, fft, heart_rates, raw, H, green_avg = pos_based_method_improved(roi_time_series, fps)

            sub1.text(x+w_steps/2, y+h_steps/2, round(bpm, 1), color=(0.0, 0.0, 0.0), fontsize=7, va='center', ha='center')
            sub2.text(int(x/w_steps), int(y/h_steps), round(bpm, 1), color=(0.745, 0.467, 0.294), fontsize=8, va='center', ha='center')
            cv2.rectangle(last_frame_clone, (x, y), (x + w_steps, y + h_steps), (0, 0, 0), 2)

            bpm_values[int(y/h_steps), int(x/w_steps)] = bpm
            skin_mat[int(y/h_steps), int(x/w_steps)] = compare_pulse_vals(bpm, pulse_lower, pulse_upper)

    # check neighbouring rois
    bool_skin_mat = eliminate_weak_skin(skin_mat)
    # save_rois_with_label(bool_skin_mat, last_frame, height, width, h_steps, w_steps, file[:-4])

    sub1.set_title('BPM on ROIs')
    sub1.imshow(last_frame_clone)
    sub2.set_title('BPM Matrix')
    sub2.matshow(bpm_values, cmap=plt.cm.gray)
    sub3.set_title('Skin, Non-Skin Matrix')
    sub3.matshow(skin_mat, cmap=plt.cm.gray)
    sub4.set_title('Skin, Neighbour reduced Matrix')
    sub4.matshow(bool_skin_mat, cmap=plt.cm.gray)

    plt.tight_layout()
    fig.savefig(file_path[:-4] + '.jpg')

    plt.close()
    # plt.show()
    # plot_results(fft, heart_rates, raw, hann_w, S, green_avg)


if __name__ == '__main__':

    start_time = time.time()
    # input_dir_path = os.path.join('assets', 'Vid_Original')
    input_dir_path = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    dest_dir_path = os.path.join('assets', 'Pulse_Data', '')
    dest_skin_dir_path = os.path.join('assets', 'Skin_Label_Data', '')

    file = '00128.MTS'
    file_path = os.path.join(input_dir_path, file)


    # pulse_label_data = load_label_data()
    #
    # # single_video_calculation(file, file_path, pulse_label_data)
    # multi_video_calculation(input_dir_path, pulse_label_data)

    skin_detection_algorithm_multi_video(input_dir_path, dest_skin_dir_path)


    print("--- %s seconds ---" % (time.time() - start_time))

