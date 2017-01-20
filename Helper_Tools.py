import os
import cv2
import numpy as np
from scipy import ndimage


def load_label_data():
    csv_file = os.path.join('assets', 'Label_Table.csv')
    f = open(csv_file, 'r')
    f.readline()

    video_nums = []
    pulse_vals_1 = []
    pulse_vals_2 = []

    for line in f:
        line = line.replace('\n', '')
        line_split = line.split(';')
        video_num = '00' + line_split[0] + '.MTS'
        video_nums.append(video_num)

        pulse_val_1 = line_split[4]
        pulse_vals_1.append(pulse_val_1)
        pulse_val_2 = line_split[5]
        pulse_vals_2.append(pulse_val_2)

    final_csv_data = np.array([video_nums, pulse_vals_1, pulse_vals_2])
    final_csv_data = np.transpose(final_csv_data)

    f.close()

    return final_csv_data


def get_pulse_vals_from_label_data(pulse_label_data, filename):

    data_index = np.where(pulse_label_data == filename)
    pulse_val_1 = pulse_label_data[data_index[0], [1]]
    pulse_val_2 = pulse_label_data[data_index[0], [2]]

    int_pulse_1 = np.asscalar(np.int16(pulse_val_1))
    int_pulse_2 = np.asscalar(np.int16(pulse_val_2))

    return int_pulse_1, int_pulse_2


def compare_pulse_vals(bpm, pulse_val_lower, pulse_val_upper):

    dist = pulse_val_upper - pulse_val_lower
    dist = 3 * np.exp(-dist / 10)

    low_bound = pulse_val_lower - dist
    up_bound = pulse_val_upper + dist

    if bpm < low_bound or bpm > up_bound:
        return 0.0
    else:
        return 28.0


def test_func(values):
    if values[4] > 0:
        return values.sum()
    else:
        return 0


def eliminate_weak_skin(skin_mat):

    footprint = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])

    results = ndimage.generic_filter(skin_mat, test_func, footprint=footprint, mode='constant')

    # 255 / 9 = 28 -> 2 neighbouring regions, threshold must be >56
    bool_skin_mat = results > 56
    return bool_skin_mat


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



if __name__ == '__main__':

    arr = np.array([[0, 0, 0, 0, 0,  28,  28,  28, 0,   0, 0,   0, 0, 0,   0, 0],
                    [0, 0, 0, 28,  28,  84,  84,  84,  56,  56,  56,  28, 0, 0, 0, 0],
                    [0, 0, 0, 28,  56, 140, 168, 168, 140, 112, 112,  56, 56, 28, 56, 28],
                    [0, 0, 0, 56, 112, 196, 224, 224, 196, 168, 168, 112, 112, 56, 84, 28],
                    [0, 0, 0, 28, 84, 168, 224, 252, 224, 196, 168, 112, 112, 56, 84, 28],
                    [28, 56, 56, 56, 84, 168, 224, 224, 168, 168, 140, 112, 56, 28, 28, 0],
                    [28, 56, 56,  28, 28, 84, 168, 168, 140, 140, 140, 112, 28, 0, 0, 0],
                    [28, 56, 56,  28, 28, 56, 112,  84,  56,  56,  84,  84, 28, 0, 0, 0]])

    eliminate_weak_skin(arr)
