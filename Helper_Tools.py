import os
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# load reference bpms from csv-file into array
def load_reference_data():
    csv_file = os.path.join('assets', 'Label_Table.csv')
    f = open(csv_file, 'r')
    f.readline()

    video_nums = []
    pulse_vals_1 = []
    pulse_vals_2 = []

    for line in f:
        line = line.replace('\n', '')
        line_split = line.split(';')
        video_num = '00' + line_split[0]
        video_nums.append(video_num)

        pulse_val_1 = line_split[4]
        pulse_vals_1.append(pulse_val_1)
        pulse_val_2 = line_split[5]
        pulse_vals_2.append(pulse_val_2)

    final_csv_data = np.array([video_nums, pulse_vals_1, pulse_vals_2])
    final_csv_data = np.transpose(final_csv_data)

    f.close()

    return final_csv_data

# get lower and upper reference bpm value from array for a certain video
def get_pulse_vals_from_label_data(pulse_label_data, filename):

    data_index = np.where(pulse_label_data == filename[:-4])
    pulse_val_1 = pulse_label_data[data_index[0], [1]]
    pulse_val_2 = pulse_label_data[data_index[0], [2]]

    int_pulse_1 = np.asscalar(np.int16(pulse_val_1))
    int_pulse_2 = np.asscalar(np.int16(pulse_val_2))

    return int_pulse_1, int_pulse_2


# compare bpms with reference bpms lower and upper
def compare_pulse_vals(bpm_map, pulse_val_lower, pulse_val_upper):

    # initialize skin-map
    skin_map = np.ones([bpm_map.shape[0], bpm_map.shape[1]], dtype='uint8')

    # calculate difference between unpper and lower reference value
    # if interval is high tolerance is lower than for smaller intervals
    dist = pulse_val_upper - pulse_val_lower
    dist = 3 * np.exp(-dist / 10)

    low_bound = pulse_val_lower - dist
    up_bound = pulse_val_upper + dist

    to_low_bpm_index = bpm_map < low_bound
    to_high_bpm_index = bpm_map > up_bound

    skin_map[to_low_bpm_index] = 0
    skin_map[to_high_bpm_index] = 0

    return skin_map


def test_func(values):
    if values[4] > 0:
        return values.sum()
    else:
        return 0

# check neighbouring BPMs and eliminate skin with less neighbouring skin-regions than thres
def eliminate_weak_skin(skin_mat, skin_neighbors=3):

    # filter
    footprint = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])

    # custom filter function
    # is not going over the boarders
    results = ndimage.generic_filter(skin_mat, test_func, footprint=footprint, mode='constant')

    # consider neighbouring regions, values to true above threshold
    bool_skin_mat = results > skin_neighbors
    return bool_skin_mat


# To get TP, FP, FN, TN for precision, recall and f_measure
def compare_with_skin_mask(file, skin_map, h_div, w_div):

    skin_label_dir = os.path.join('assets', 'Skin_Label_Data')
    skin_mask_path = os.path.join(skin_label_dir, 'Skin_Masks_' + str(h_div) + 'x' + str(w_div), 'Skin_' + str(h_div)
                                  + 'x' + str(w_div) + '_' + file[:-4] + '.npy')
    print(skin_mask_path)
    skin_mask = np.load(skin_mask_path)
    diff_map = skin_mask - skin_map

    false_negatives = np.count_nonzero(diff_map == 1)
    false_positives = np.count_nonzero(diff_map == -1)
    positives = np.count_nonzero(skin_mask == 1)
    negatives = np.count_nonzero(skin_mask == 0)
    true_positives = positives - false_negatives
    true_negatives = negatives - false_positives

    precision, recall, f_measure, support = precision_recall_fscore_support(skin_mask, skin_map, average='micro')

    print('Precision: ' + str(precision*100) + '%', 'Recall: ' + str(recall*100) + '%', 'F-Measure: ' + str(f_measure*100) + '%')

    return true_positives, false_positives, false_negatives, true_negatives


# Old function to save small roi images with skin or non-skin label
def save_rois_with_label(bool_skin_mat, frame, height, width, h_steps, w_steps, file):

    dest_folder = os.path.join('assets', 'ROIs', file, '')
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for x in range(0, width, w_steps):
        for y in range(0, height, h_steps):
            roi = frame[y:y + h_steps, x:x + w_steps]

            i = int(y / h_steps)
            j = int(x / w_steps)

            if bool_skin_mat[i, j]:
                file_path_out = dest_folder + 'skin_' + file + '_' + str(j) + str(i) + '.png'
                cv2.imwrite(file_path_out, roi)
            else:
                file_path_out = dest_folder + 'non_skin_' + file + '_' + str(j) + str(i) + '.png'
                cv2.imwrite(file_path_out, roi)

