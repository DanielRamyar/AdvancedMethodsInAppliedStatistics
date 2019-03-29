import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = 'digit-recognizer/train.csv'
data = pd.read_csv(file)


# print(data.head())
# print(data.iloc[0, 1::])


numpy_data = data.values
data_mean_vector = np.mean(numpy_data[:, 1:], axis=0)

data_tilte = (numpy_data[:, 1:] - data_mean_vector)
cov_mat = np.cov(data_tilte.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key = lambda x: x[0], reverse= True)


image = numpy_data[0, 1:].reshape((28, 28))
image1 = data_tilte[0, :].reshape((28, 28))

plt.imshow(image, cmap='hot')
plt.colorbar()

plt.figure()

plt.imshow(image1, cmap='hot')
plt.colorbar()
plt.figure()

plt.imshow(eig_pairs[1][1].reshape((28, 28)), cmap='hot')
plt.colorbar()
plt.show()