import numpy as np
import os
from matplotlib import pyplot as plt

# pulse_signal_dataset_path = os.path.join('assets', 'ROI_Full_Dataset.npy')
# pulse_signal_dataset_path = os.path.join('assets', 'Full_Dataset.npy')
# pulse_signal_dataset_path = os.path.join('assets', 'Balanced_00130.npy')
# pulse_signal_dataset_path = os.path.join('assets', 'Pulse_Data', '00130.npy')
pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'ROIs', 'Balanced_ROI_00132.npy')
roi = os.path.join('assets', 'Pulse_Data', 'ROIs', 'ROI_00163.npy')
server_roi = os.path.join('assets', 'Pulse_Data', 'ROIs', 'Server_ROI_00163.npy')


def compare_two_npys(npy_path_1, npy_path_2):

    npy_1 = np.load(npy_path_1)
    npy_2 = np.load(npy_path_2)

    npy_1_f32 = npy_1.astype('float32')
    npy_2_f32 = npy_2.astype('float32')

    is_equal = np.array_equal(npy_1_f32, npy_2_f32)

    print('Arrays are equal: ' + str(is_equal))


def get_bpm_from_pruned_fft(pruned_ffts):
    fps = 25
    L = 257

    max_freq_pos = np.argmax(pruned_ffts, axis=1)

    frequencies = np.linspace(0, fps / 2, L, endpoint=True)
    heart_rates = frequencies * 60
    human_rates = heart_rates[17:61]

    bpms = human_rates[max_freq_pos]

    return bpms


def check_signal_values():
    print("Loading " + pulse_signal_dataset_path)
    dataset = np.load(pulse_signal_dataset_path)

    # X = dataset.reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2])

    X = dataset[:, 0:44]
    Y = dataset[:, 44]

    bpms = get_bpm_from_pruned_fft(X)


    print(Y)

    print(bpms)

    # bpms = bpms.reshape(32, 64)
    # plt.matshow(bpms, cmap=plt.cm.gray)
    # plt.show()


if __name__ == '__main__':

    # check_signal_values()
    compare_two_npys(roi, server_roi)
