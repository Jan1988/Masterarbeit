import numpy as np
import os


width = 5
puls_signal_data = np.zeros([width], dtype='float64')
for x in range(0, width):
    puls_signal_data[x] = 2



in_img = os.path.join('Assets', 'Mila_Ass_2.jpg')
out_csv = os.path.join('Assets', 'Test.txt')

data = np.zeros([8, 10], dtype='float64')

H2 = np.arange(10).reshape((1, 10))
I2 = np.arange(10).reshape((1, 10))

H = np.ones([1, 10], dtype='float64')
I = np.ones([1, 10], dtype='float64')


data[7, :] = H2
data[1, :] = I2

# Write the array to disk
with open(out_csv, 'wb') as outfile:

    np.savetxt(outfile, data, fmt='%i')


# Read the array from disk
new_data = np.loadtxt('assets\Test.txt')

print(new_data)

new_data = new_data.reshape((2, 4, 10))

print(type(new_data))
print(new_data)
