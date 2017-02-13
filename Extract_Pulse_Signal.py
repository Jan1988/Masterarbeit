
import os
import time

import cv2
import numpy as np
import scipy.misc

from matplotlib import pyplot as plt

from Video_Tools import load_video, get_video_dimensions
from POS_Based_Method import extract_pos_based_method_improved
from Helper_Tools import load_label_data, get_pulse_vals_from_label_data, compare_pulse_vals, eliminate_weak_skin, \
    compare_with_skin_mask

true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0


# Plot for Thesis Image Pixelwise
def plot_and_save_results(_plot_title, last_frame, _bpm_map, _weak_skin_map, _strong_skin_map, save_to):

    bgr_frame = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)

    fig = plt.figure(figsize=(19, 15))
    fig.suptitle(_plot_title, fontsize=25, fontweight='bold')
    tick_fontsize = 15
    txt_coord_x = 0.05
    txt_coord_y = 0.9
    txt_fontsize = 21

    sub1 = fig.add_subplot(221)
    sub2 = fig.add_subplot(222)
    sub3 = fig.add_subplot(223)
    sub4 = fig.add_subplot(224)

    sub1.text(txt_coord_x, txt_coord_y, '(a)', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub1.transAxes)
    sub1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub1.imshow(bgr_frame)

    sub2.text(txt_coord_x, txt_coord_y, '(b)', color='white', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub2.transAxes)
    sub2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub2.matshow(_bpm_map, cmap=plt.cm.gray)

    sub3.text(txt_coord_x, txt_coord_y, '(c)', color='white', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub3.transAxes)
    sub3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub3.matshow(_weak_skin_map, cmap=plt.cm.gray)

    sub4.text(txt_coord_x, txt_coord_y, '(d)', color='white', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub4.transAxes)
    sub4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub4.matshow(_strong_skin_map, cmap=plt.cm.gray)

    plt.tight_layout()
    # plt.show()

    fig.savefig(save_to + '.png')
    plt.close()
    # scipy.misc.imsave(out_file_path[:-4] + '.png', bpm_map)


def extr_multi_video_calculation(in_dir, out_dir, roi=False):
    for file in os.listdir(in_dir):
        in_file_path = os.path.join(in_dir, file)

        if file.endswith(".mkv"):
            if roi:
                print('This is a ROI Calculation')
                extr_roi_single_video_calculation(file, in_file_path, out_dir)
            else:
                print('This is a Pixelwise Calculation')
                extr_single_video_calculation(file, in_file_path, out_dir)


# For ROI
def extr_roi_single_video_calculation(in_file, in_file_path, out_dir):

    w_div = 16
    h_div = 8

    video_frames, fps = load_video(in_file_path)
    video_frames = video_frames[22:358]
    frame_count, width, height = get_video_dimensions(video_frames)
    w_steps = int(width / w_div)
    h_steps = int(height / h_div)

    # Giant-ndarray for pulse-signals for height*width of a Videos
    pulse_signal_data = np.zeros([h_div, w_div, 44], dtype='float64')
    bpm_map = np.zeros((h_div, w_div), dtype='float64')

    # Load all pulse value belonging to a certain video in array
    # pulse_lower, pulse_upper = get_pulse_vals_from_label_data(load_label_data(), in_file)
    pulse_upper = 52
    pulse_lower = 46
    fps = 25

    # For plotting the skin and pulse matrices
    plot_title = file + ' BPM: ' + str(pulse_lower) + '-' + str(pulse_upper)
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(plot_title, fontsize=20, fontweight='bold')
    tick_fontsize = 11
    txt_coord_x = 0.05
    txt_coord_y = 0.9
    txt_fontsize = 21

    sub1 = fig.add_subplot(221)
    sub2 = fig.add_subplot(222)
    sub3 = fig.add_subplot(223)
    sub4 = fig.add_subplot(224)

    last_frame = video_frames[frame_count - 1]
    last_frame_clone = last_frame.copy()

    # The odd rest is cutted here
    width = w_steps * w_div
    height = h_steps * h_div
    for x in range(0, width, w_steps):
        for y in range(0, height, h_steps):
            roi_ind_x = int(x / w_steps)
            roi_ind_y = int(y / h_steps)

            roi_time_series = video_frames[:, y:y+h_steps, x:x+w_steps]
            # Spatial Averaging
            roi_time_series_avg = np.mean(roi_time_series, axis=(1, 2))

            bpm, pruned_fft = extract_pos_based_method_improved(roi_time_series_avg, fps)

            # for validating if extr method is the same as main method
            bpm_map[roi_ind_y, roi_ind_x] = bpm

            pulse_signal_data[roi_ind_y, roi_ind_x] = pruned_fft

            # For plotting the skin and pulse matrices
            sub1.text(x + w_steps / 2, y + h_steps / 2, round(bpm, 1), color=(0.0, 0.0, 0.0), fontsize=7, va='center', ha='center')
            cv2.rectangle(last_frame_clone, (x, y), (x + w_steps, y + h_steps), (0, 0, 0), 2)

        print("Fortschritt: %.2f %%" % ((x + 1.0) / width * 100.0))

    # check neighbouring BPMs
    weak_skin_map = compare_pulse_vals(bpm_map, pulse_lower, pulse_upper)
    strong_skin_map = eliminate_weak_skin(weak_skin_map, skin_neighbors=3)

    out_file_path = os.path.join(out_dir, 'me_' + in_file[:-4])
    bgr_last_frame = cv2.cvtColor(last_frame_clone, cv2.COLOR_RGB2BGR)
    # For plotting the skin and pulse matrices
    sub1.text(txt_coord_x, txt_coord_y, '(a)', color='white', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub1.transAxes)
    sub1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub1.imshow(bgr_last_frame)
    sub2.text(txt_coord_x, txt_coord_y, '(b)', color='white', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub2.transAxes)
    sub2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub2.matshow(bpm_map, cmap=plt.cm.gray)
    sub3.text(txt_coord_x, txt_coord_y, '(c)', color='white', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub3.transAxes)
    sub3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub3.matshow(weak_skin_map, cmap=plt.cm.gray)
    sub4.text(txt_coord_x, txt_coord_y, '(d)', color='white', fontsize=txt_fontsize, horizontalalignment='center',
              transform=sub4.transAxes)
    sub4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    sub4.matshow(strong_skin_map, cmap=plt.cm.gray)

    plt.tight_layout()
    # plt.show()
    fig.savefig(out_file_path + '.png')
    plt.close()

    # # POS Metrics measure
    # vid_true_positives, vid_false_positives, vid_false_negatives, vid_true_negatives = compare_with_skin_mask(file, weak_skin_map, h_div, w_div)
    # global true_positives
    # global false_positives
    # global false_negatives
    # global true_negatives
    # true_positives += vid_true_positives
    # false_positives += vid_false_positives
    # false_negatives += vid_false_negatives
    # true_negatives += vid_true_negatives

    print("--- File Completed after %s seconds ---" % (time.time() - start_time))
    np.save(out_file_path, pulse_signal_data)
    print('Saved to ' + out_file_path)


# For Pixelwise
def extr_single_video_calculation(in_file, in_file_path, out_dir):

    video_frames, fps = load_video(in_file_path)
    video_frames = video_frames[22:358]
    frame_count, width, height = get_video_dimensions(video_frames)


    # Giant-ndarray for pulse-signals for height*width of a Videos
    pulse_signal_data = np.zeros([height, width, 44], dtype='float64')
    # Only for visualizing
    bpm_map = np.zeros([height, width], dtype='float16')

    # # Do this on server
    # video_frames = video_frames.astype('float32')
    # video_frames += 1.0

    # For plotting the skin and pulse matrices
    last_frame = video_frames[frame_count - 1]
    last_frame_clone = last_frame.copy()
    # Load all pulse value belonging to a certain video in array
    pulse_lower, pulse_upper = get_pulse_vals_from_label_data(load_label_data(), in_file)

    for x in range(0, width):
        for y in range(0, height):

            px_time_series = video_frames[:, y, x]

            bpm, pruned_fft = extract_pos_based_method_improved(px_time_series, fps)

            pulse_signal_data[y, x] = pruned_fft

            bpm_map[y, x] = bpm

        print("Completed: %.2f %%" % ((x + 1.0) / width * 100.0))

    # check neighbouring BPMs
    weak_skin_map = compare_pulse_vals(bpm_map, pulse_lower, pulse_upper)
    strong_skin_map = eliminate_weak_skin(weak_skin_map, skin_neighbors=5)

    out_file_path = os.path.join(out_dir, 'no_nan_' + in_file[:-4])

    # For plotting the skin and pulse matrices
    plot_title = in_file + ' BPM: ' + str(pulse_lower) + '-' + str(pulse_upper)
    plot_and_save_results(plot_title, last_frame_clone, bpm_map, weak_skin_map, strong_skin_map, out_file_path)

    np.save(out_file_path, pulse_signal_data)
    print("--- File Completed after %s seconds ---" % (time.time() - start_time))
    print('Saved to ' + out_file_path)


if __name__ == '__main__':

    start_time = time.time()
    file = '00081.mkv'
    Pulse_data_dir = os.path.join('assets', 'Pulse_Data')
    video_dir_me = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht', 'Me')
    video_dir = os.path.join('assets', 'Vid_Original', 'Kuenstliches_Licht')
    video_file_path = os.path.join(video_dir_me, file)

    extr_single_video_calculation(file, video_file_path, Pulse_data_dir)

    # extr_multi_video_calculation(video_dir_me, Pulse_data_dir, roi=True)
    # extr_multi_video_calculation(video_dir_me, Pulse_data_dir, roi=False)

    # extr_roi_single_video_calculation(file, video_file_path, Pulse_data_dir)

    print(true_positives, false_positives, false_negatives, true_negatives)
    print("--- Algorithm Completed %s seconds ---" % (time.time() - start_time))


