import os
import time
import cv2
import numpy as np

from matplotlib import pyplot as plt

from Video_Tools import load_video, get_video_dimensions
from CHROM_Based_Method import chrom_based_pulse_signal_estimation
from POS_Based_Method import pos_based_method_improved, extract_pos_based_method_improved
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin, \
    save_rois_with_label
from Skin_Mask_Creation import skin_detection_algorithm_multi_video

start_time = time.time()
input_dir_path_nat = os.path.join('assets', 'Vid_Original', 'Natuerliches_Licht')
input_dir_path_kue = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
dest_dir_path = os.path.join('assets', 'Pulse_Data', '')
dest_skin_dir_path = os.path.join('assets', 'Skin_Label_Data', '')



# Plot for Thesis Image
def plot_results(bpm, fft2, fft1, heart_rates, raw=0, overlap_signal=0, pulse_signal=0, norm_channels=0, time_series=0):
    # plt.axis([0, n, y_lower, y_upper])
    tick_fontsize = 7
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

    # plt.tight_layout()
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


def single_video_calculation(file, file_path, pulse_label_data):
    start_time = time.time()
    w_div = 64
    h_div = 32

    skin_mat = np.zeros((h_div, w_div), dtype='float64')
    bpm_values = np.zeros((h_div, w_div), dtype='float64')

    video_frames, fps = load_video(file_path)
    video_frames = video_frames[22:310]
    frame_count, width, height = get_video_dimensions(video_frames)
    roi_width = int(width / w_div)
    roi_height = int(height / h_div)

    # Load all pulse value belonging to a certain video in array
    # Will be used for ROI labeling
    pulse_lower, pulse_upper = get_pulse_vals_from_label_data(pulse_label_data, file)

    # Fuer die Darstellung der Puls Ergebnismatrix
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(file, fontsize=14, fontweight='bold')
    sub1 = fig.add_subplot(221)
    sub2 = fig.add_subplot(222)
    sub3 = fig.add_subplot(223)
    sub4 = fig.add_subplot(224)

    last_frame = video_frames[frame_count - 1]
    last_frame_clone = last_frame.copy()

    '''BPM Estimation for every ROI'''

    # Hier wird der ungeradere Rest abgeschnitten
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
            # bpm, pruned_fft, fft, heart_rates, raw, H, h, norm_channels, time_series = pos_based_method_improved(time_series, fps)
            bpm, pruned_fft = extract_pos_based_method_improved(time_series, fps)

            sub1.text(x + roi_width / 2, y + roi_height / 2, round(bpm, 1), color=(0.0, 0.0, 0.0), fontsize=5,
                      va='center', ha='center')
            # sub2.text(roi_ind_x, roi_ind_y, round(bpm, 1), color=(0.745, 0.467, 0.294), fontsize=5, va='center',
            #           ha='center')
            cv2.rectangle(last_frame_clone, (x, y), (x + roi_width, y + roi_height), (0, 0, 0), 2)
            # plot_results(bpm, pruned_fft, fft, heart_rates, raw=raw, overlap_signal=H, pulse_signal=h, norm_channels=norm_channels, time_series=time_series)

            bpm_values[roi_ind_y, roi_ind_x] = bpm
            skin_mat[roi_ind_y, roi_ind_x] = compare_pulse_vals(bpm, pulse_lower, pulse_upper)
        print("Fortschritt: %.2f %%" % ((x+1.0) / width*100.0))


    # check neighbouring rois
    bool_skin_mat = eliminate_weak_skin(skin_mat)
    # save_rois_with_label(bool_skin_mat, last_frame, height, width, roi_height, roi_width, file[:-4])

    # Fuer die Darstellung der Puls Ergebnismatrix
    sub1.set_title('BPM on ROIs')
    sub1.imshow(last_frame_clone)
    sub2.set_title('BPM Matrix')
    sub2.matshow(bpm_values, cmap=plt.cm.gray)
    sub3.set_title('Skin, Non-Skin Matrix')
    sub3.matshow(skin_mat, cmap=plt.cm.gray)
    sub4.set_title('Skin, Neighbour reduced Matrix')
    sub4.matshow(bool_skin_mat, cmap=plt.cm.gray)
    plt.tight_layout()
    fig.savefig(file_path[:-4] + '.png')
    plt.close()

    # plt.show()



if __name__ == '__main__':

    file = '00132.MTS'
    file_path = os.path.join(input_dir_path_kue, file)
    pulse_label_data = load_label_data()

    # single_video_calculation(file, file_path, pulse_label_data)

    multi_video_calculation(input_dir_path_kue, pulse_label_data)

    # skin_detection_algorithm_multi_video(input_dir_path, dest_skin_dir_path)


    print("--- Algorithm Completed %s seconds ---" % (time.time() - start_time))
