import numpy as np
import os
from random import randrange
import sklearn

file = '00130.npy'
signal_data_path = os.path.join('assets', file)
skin_mask_data_path = os.path.join('assets', 'Skin_' + file)
out_balanced_signal_data = os.path.join('assets', 'Balanced_' + file)


# signal_data = np.ndarray((pixel_count, 44, width, height), dtype=np.float64)

signal_data = np.load(signal_data_path)


skin_mask_data = np.load(skin_mask_data_path)

print(signal_data.shape)
print(skin_mask_data.shape)

# Where values are low
skin_indices = skin_mask_data > 0
non_skin_indices = skin_mask_data < 1
skin_count = len(skin_mask_data[skin_indices])
non_skin_count = len(skin_mask_data[non_skin_indices])

print('Count of Skin Samples: ' + str(skin_count))
print('Count of Non-Skin Samples: ' + str(non_skin_count))

one_labels = np.ones((skin_count, 1))
zero_labels = np.zeros((skin_count, 1))

skin_signal_data = signal_data[skin_indices, :]
non_skin_signal_data = signal_data[non_skin_indices, :]

print(skin_signal_data.shape)
print(non_skin_signal_data.shape)

random_choice = np.random.choice(non_skin_count, size=skin_count, replace=False)

subsampled_non_skin_signal_data = non_skin_signal_data[random_choice, :]

final_skin_signal_data = np.concatenate((skin_signal_data, one_labels), axis=1)
final_non_skin_signal_data = np.concatenate((subsampled_non_skin_signal_data, zero_labels), axis=1)

print('Skin Samples shape: ' + str(final_skin_signal_data.shape))
print('Non-Skin Samples shape: ' + str(final_non_skin_signal_data.shape))

print(final_skin_signal_data[150000, 44])
print(final_non_skin_signal_data[150000, 44])

balanced_signal_data = np.concatenate((final_skin_signal_data, final_non_skin_signal_data))

print(balanced_signal_data.shape)

print(balanced_signal_data[373392, 44])
print(balanced_signal_data[373393, 44])

np.save(out_balanced_signal_data, balanced_signal_data)
print('Saving: ' + out_balanced_signal_data)
