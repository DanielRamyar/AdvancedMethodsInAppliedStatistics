import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def display_digit(data, index=None):
    '''
    Plots the digit on given index of data. If index is not given it assummes
    a single line a data.
    '''
    if index is not None:
        plt.imshow(data[index, 1::].reshape((28, 28)))
    else:
        plt.imshow(data[1::].reshape((28, 28)))


def init_centroids(data, n):
    """
    Returns random initial centroids from data.
    """
    index = np.random.randint(len(data), size=(n))
    init_cent = data[index, :]
    return init_cent


def dist_centroids(data, cent):

    A = []
    for i in range(cent.shape[0]):
        temp = data[:, 1::] - cent[i, 1::]
        temp2 = np.square(temp)
        temp3 = np.sum(temp2, axis=1)
        temp4 = np.sqrt(temp3)
        A = np.append(A, temp4)

    A = A.reshape((cent.shape[0], data.shape[0])).T
    return A


# Read file
file_path = 'digit-recognizer/train.csv'
data = pd.read_csv(file_path, nrows=100)
test = data.values

print(data.iloc[0].values[1::].shape)
print(test[0, 1::].shape)
initc = init_centroids(test, 2)
print(dist_centroids(test, initc).shape)

# display_digit(data.iloc[0].values)
# plt.figure()
# display_digit(test, 0)
# plt.show()


# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.2,
#                                                     random_state=42)

# nr_unique_digits = len(np.unique(y_train))

# kmeans = KMeans(n_clusters=50, random_state=0).fit(X_train)
# predict_kmeans = kmeans.predict(X_test)

# print(kmeans.labels_[0:20])
# for i in range(10):
#     plt.figure()
#     plt.imshow(X_train[i, :].reshape((28, 28)))


