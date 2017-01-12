

import numpy as np


time_series = np.array([[109, 139, 164], [107, 140, 164], [106, 139, 163], [105, 139, 161], [105, 139, 161],
                     [106, 140, 162], [105, 139, 161], [102, 136, 158], [105, 140, 159], [106, 141, 160],
                     [107, 141, 163], [107, 141, 163], [107, 141, 163], [107, 142, 161], [107, 142, 161], [102, 136, 158]])

# sliding window size
window_size = 4
sequence_length = len(time_series)
interval_count = int(sequence_length/window_size)*2

H = np.zeros(sequence_length, dtype='float64')

n = window_size
for i in range(interval_count):

    m = n - window_size

    # 5 temporal normalization
    window = time_series[m:n]
    mean_array = np.average(window, axis=0)
    norm_array = window / mean_array

    # 6 projection
    S1 = np.dot(norm_array, [-1, 1, 0])
    S2 = np.dot(norm_array, [1, 1, -2])

    # 7 tuning
    S1_std = np.std(S1)
    S2_std = np.std(S2)

    alpha = S1_std / S2_std

    h = S1 + alpha * S2

    # # make h zero-mean
    # h_mean = np.mean(h)
    # h_no_mean = h - h_mean

    # Hann window signal
    hann_window = np.hanning(len(h))
    hann_windowed_signal = hann_window * h

    H[m:n] += hann_windowed_signal
    print(H)

    n += int(window_size/2)



