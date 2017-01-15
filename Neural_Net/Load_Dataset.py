#import pandas as pd
#
# # Read the array from disk
# # new_data = np.loadtxt(file_path_out)
# new_data = pd.read_csv(file_path_out, sep=" ", header=None)
#
# # original shape of the array
# new_data = new_data.values
#
# reshaped_new_data = new_data.reshape((3, 1920, 44))
#
# test_val1 = reshaped_new_data[0, 0]
# test_val2 = pulse_signal_data[0, 0]