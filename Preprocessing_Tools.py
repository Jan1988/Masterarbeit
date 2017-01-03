import os
import numpy as np


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
    dist = 5 * np.exp(-dist / 10)

    low_bound = pulse_val_lower - dist
    up_bound = pulse_val_upper + dist

    if bpm < low_bound or bpm > up_bound:
        return 255.0
    else:
        return 100.0


if __name__ == '__main__':
    final_csv = load_label_data()

    pulse_val_1, pulse_val_2 = get_pulse_vals_from_label_data(final_csv, '00100.MTS')

    print(pulse_val_1)
    print(pulse_val_2)
