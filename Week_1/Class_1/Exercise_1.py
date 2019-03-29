import numpy as np

file_name = 'FranksNumbers.txt'

my_file = open(file_name, 'r')

my_array = np.zeros((1,2))
print(my_array)

counter = -1
row_counter = 0

for line in my_file:

    current_line = line.split()
    if len(current_line) > 3:
        pass
    elif len(current_line) == 3:
        counter += 1
        my_array = np.expand_dims(my_array, axis = -1)
        print(my_array.shape)
    elif len(current_line) == 0:
        print('Empty Line!')
    else:
        temp_array = np.array([float(current_line[0]), float(current_line[1])])
        temp_array = np.expand_dims(temp_array, axis = 0)
        print(temp_array)
        print(my_array[:,:,0].shape)
        print(counter)
        my_array[:, :, counter] = np.stack((my_array[:, :, counter], temp_array))


my_array = np.delete(my_array, 0, 0)
print(my_array)
