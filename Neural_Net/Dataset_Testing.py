import numpy as np
import os
from matplotlib import pyplot as plt

# pulse_signal_dataset_path = os.path.join('assets', 'ROI_Full_Dataset.npy')
# pulse_signal_dataset_path = os.path.join('assets', 'Full_Dataset.npy')
# pulse_signal_dataset_path = os.path.join('assets', 'Balanced_00130.npy')
# pulse_signal_dataset_path = os.path.join('assets', 'Pulse_Data', '00130.npy')
# pulse_signal_dataset_path = os.path.join('assets', 'Balanced_Data', 'ROIs', 'Balanced_ROI_00132.npy')
pulse_signal_dataset_path = os.path.join('assets', 'Pulse_Data', 'ROIs', 'ROI_00132.npy')

def check_signal_values():
    print("Loading " + pulse_signal_dataset_path)
    dataset = np.load(pulse_signal_dataset_path)

    # bpms = np.zeros(len(X))
    for i, row in enumerate(dataset):
        print(row)

    # X = dataset.reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2])

    fps = 25
    L = 257

    # X = dataset[:, 0:44]
    # Y = dataset[:, 44]

    max_freq_pos = np.argmax(X, axis=1)

    frequencies = np.linspace(0, fps / 2, L, endpoint=True)
    heart_rates = frequencies * 60
    human_rates = heart_rates[17:61]


    half_samples = int(len(X) / 2)



    #
    # print(X[1079])
    # print(X[1080])
    # print(X[1081])
    # print(X[1082])

    bpms = human_rates[max_freq_pos]

    # for i, max_pos in enumerate(max_freq_pos):
    #
    #     bpm = human_rates[max_pos]

        # print(bpm, Y[i])
        # print(human_rates(max_freq_pos[half_samples + i]), Y[half_samples + i])

    # low_index = bpms < 57
    # high_index = bpms > 67
    # bpms[low_index] = 0
    # bpms[high_index] = 0
    #
    # bpms = bpms.reshape(32, 64)
    # plt.matshow(bpms, cmap=plt.cm.gray)
    # plt.show()


if __name__ == '__main__':

    check_signal_values()

