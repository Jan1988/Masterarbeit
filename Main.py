import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Helper_Tools import load_reference_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin
from POS_Based_Method import extract_pos_based_method_improved, pos_based_method_improved
from Video_Tools import load_video, get_video_dimensions


start_time = time.time()
input_dir_path_nat = os.path.join('assets', 'Vid_Original', 'Natuerliches_Licht')
input_dir_path_kue = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
dest_dir_path = os.path.join('assets', 'Pulse_Data', '')
dest_skin_dir_path = os.path.join('assets', 'Skin_Label_Data', '')


# Plot for Thesis Image
def plot_results(bpm, fft2, fft1, heart_rates, raw=0, overlap_signal=0, pulse_signal=0, norm_channels=0, time_series=0):
    # plt.axis([0, n, y_lower, y_upper])
    tick_fontsize = 12
    txt_fontsize = 10
    nbins_x = 8
    nbins_y = 5
    txt_coord_x = 0.05
    txt_coord_y = 0.9
    txt_fontsize = 17

    fig = plt.figure(figsize=(19, 15))
    fig.suptitle('BPM: ' + str(bpm), fontsize=20, fontweight='bold')

    sub1 = fig.add_subplot(321)

    sub1.text(txt_coord_x, txt_coord_y, '(a)', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub1.transAxes)
    # sub1.set_title('Normalized RGB-Channels', fontsize=subtitle_fontsize, loc='bottom')
    sub1.plot(norm_channels[:, 2], 'r',
              norm_channels[:, 1], 'g',
              norm_channels[:, 0], 'b')
    sub1.set_ylim([np.amin(norm_channels) * 0.9975, np.amax(norm_channels) * 1.0025])
    sub1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # to specify number of ticks on both or any single axes
    sub1.locator_params(axis='y', tight=True, nbins=nbins_y)
    sub1.locator_params(axis='x', nbins=nbins_x)
    sub1.ticklabel_format(useOffset=False)

    # sub1.plot(green_norm, 'g')

    sub2 = fig.add_subplot(322)
    sub2.text(txt_coord_x, txt_coord_y, '(b)', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub2.transAxes)
    # sub2.set_title('Pulse Signal', fontsize=subtitle_fontsize, )
    sub2.plot(pulse_signal, 'k')
    sub2.set_ylim([np.amin(pulse_signal) * 1.25, np.amax(pulse_signal) * 1.25])
    sub2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub2.locator_params(axis='y', tight=True, nbins=nbins_y)

    sub5 = fig.add_subplot(323)
    sub5.text(txt_coord_x, txt_coord_y, '(c)', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub5.transAxes)
    # sub5.set_title('Overlap-Added Signal', fontsize=subtitle_fontsize)
    sub5.plot(overlap_signal, 'k')
    sub5.set_ylim([np.amin(overlap_signal) * 1.25, np.amax(overlap_signal) * 1.25])
    sub5.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub5.locator_params(axis='y', tight=True, nbins=nbins_y)

    sub8 = fig.add_subplot(324)
    sub8.text(txt_coord_x, txt_coord_y, '(d)', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub8.transAxes)
    # sub8.set_title('Unverarbeitete FFT', fontsize=subtitle_fontsize)
    sub8.plot(raw, 'k')
    sub8.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    sub4 = fig.add_subplot(325)
    sub4.text(txt_coord_x, txt_coord_y, '(e)', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub4.transAxes)
    # sub4.set_title('FFT Absolut', fontsize=subtitle_fontsize)
    sub4.plot(fft1, 'k')
    sub4.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # sub6 = fig.add_subplot(336)
    # sub6.set_title('FFT pruned', fontsize=subtitle_fontsize)
    # sub6.plot(fft2, 'k')
    # sub6.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    sub7 = fig.add_subplot(326)
    sub7.text(txt_coord_x, txt_coord_y, '(f)', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub7.transAxes)
    # sub7.set_title('FFT im Herzfrequenzbereich', fontsize=subtitle_fontsize)
    sub7.plot(heart_rates, fft2, 'k')
    sub7.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.tight_layout()
    fig.savefig(file_path[:-4] + '.png', bbox_inches='tight')
    # fig.savefig(file_path[:-4] + '.png')
    plt.show()
    # plt.close()


def multi_video_calculation(dir_path, pulse_label_data):
    for file in os.listdir(dir_path):
        if file.endswith(".MTS"):
            file_path = os.path.join(dir_path, file)
            print(file_path)
            single_video_calculation(file, file_path, pulse_label_data)


# Old script - function was only used for plotting the signale curves of POS, fft, purned_fft, ...
def single_video_calculation(file, file_path, pulse_label_data):
    start_time = time.time()
    w_div = 16
    h_div = 8

    bpm_values = np.zeros((h_div, w_div), dtype='float64')

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[22:310]
    frame_count, width, height = get_video_dimensions(video_frames)
    roi_width = int(width / w_div)
    roi_height = int(height / h_div)


    width = roi_width * w_div
    height = roi_height * h_div
    for x in range(0, width, roi_width):
        for y in range(0, height, roi_height):
            roi_ind_x = int(x / roi_width)
            roi_ind_y = int(y / roi_height)

            roi_time_series = video_frames[:, y:y + roi_height, x:x + roi_width]
            # Spatial Averaging used when ROIs are extracted
            time_series = np.mean(roi_time_series, axis=(1, 2))

            # Pulse-Signal Extraction
            bpm, pruned_fft, fft, heart_rates, raw, H, h, norm_channels, time_series = pos_based_method_improved(time_series, fps)
            # bpm, pruned_fft = extract_pos_based_method_improved(time_series, fps)

            plot_results(bpm, pruned_fft, fft, heart_rates, raw=raw, overlap_signal=H, pulse_signal=h, norm_channels=norm_channels, time_series=time_series)

            bpm_values[roi_ind_y, roi_ind_x] = bpm

        print("Fortschritt: %.2f %%" % ((x+1.0) / width*100.0))


if __name__ == '__main__':

    file = '00130.MTS'
    file_path = os.path.join(input_dir_path_kue, file)
    pulse_label_data = load_reference_data()

    # single_video_calculation(file, file_path, pulse_label_data)

    multi_video_calculation(input_dir_path_kue, pulse_label_data)

    # skin_detection_algorithm_multi_video(input_dir_path, dest_skin_dir_path)

    print("--- Algorithm Completed %s seconds ---" % (time.time() - start_time))
