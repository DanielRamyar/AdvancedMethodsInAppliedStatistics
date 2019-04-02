import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import time


# I stole this function from the interwebz
def loadlocal_mnist(images_path, labels_path):
    """ Read MNIST from ubyte files.
    Parameters
    ----------
    images_path : str
        path to the test or train MNIST ubyte file
    labels_path : str
        path to the test or train MNIST class labels file
    Returns
    --------
    images : [n_samples, n_pixels] numpy.array
        Pixel values of the images.
    labels : [n_samples] numpy array
        Target class labels
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
    """
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def display_digit(data, index=None):
    '''
    Plots the digit on given index of data. If index is not given it assummes
    a single line a data.
    '''
    if index is not None:
        plt.imshow(data[index, 2::].reshape((28, 28)))
    else:
        plt.imshow(data[2::].reshape((28, 28)))


def init_centroids(data, n):
    """
    Returns random initial centroids from data.
    """
    index = np.random.randint(len(data), size=(n))
    init_cent = data[index, :]
    return init_cent


def dist_centroids(data, cent):
    '''
    Returns the distance between each point (each row in data) and centroids
    '''
    dist = []
    for i in range(cent.shape[0]):
        temp = np.linalg.norm(data[:, 2::] - cent[i, 2::], axis=1)
        dist = np.append(dist, temp)

    dist = dist.reshape((cent.shape[0], data.shape[0])).T
    return dist


def label_centroids(data, dist):
    '''
    Adds Label to data according to its closest centroid
    '''
    new_data = np.copy(data)
    temp = np.argmin(dist, axis=1) + 1
    new_data[:, 0] = temp
    return new_data


def new_centroids(data, cent):
    '''
    Calculates the new centroids from data with same label
    '''
    new_cent = np.copy(cent)
    for i in range(cent.shape[0]):
        new = np.average(data[data[:, 0] == i + 1][:, 2::], axis=0)
        new_cent[i, 2::] = new

    return new_cent


def label_clusters(data, cent, n):
    '''
    Finds most common label in cluster and sets cluster label
    '''
    for i in range(n):
        temp = data[data[:, 0] == i + 1]

        temp1 = temp[:, 1].astype(int)
        label = np.bincount(temp1).argmax()
        cent[i, 0] = label

    return cent


def classify(to_be_classified, cent):

    dist = []
    for i in range(cent.shape[0]):
        temp = np.linalg.norm(to_be_classified[:, 2::] - cent[i, 2::], axis=1)
        dist = np.append(dist, temp)
    dist = dist.reshape((cent.shape[0], to_be_classified.shape[0])).T
    temp = np.argmin(dist, axis=1)

    to_be_classified[:, 0] = cent[temp, 0]

    return to_be_classified


def get_accuracy(x, y):

    correct_classified = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            correct_classified += 1
    accuracy = correct_classified / len(x)

    return accuracy


def K_means(data, n):
    '''
    Does k-means analysis
    '''
    previous_difference = 0
    centroids = init_centroids(data, n)

    while True:
        print(1)
        distance_from_centroids = dist_centroids(data, centroids)
        data = label_centroids(data, distance_from_centroids)
        centroids_new = new_centroids(data, centroids)

        difference = centroids[:, 2::] - centroids_new[:, 2::]
        norm_difference = np.linalg.norm(difference, axis=1)

        largest_norm_difference = max(norm_difference)
        difference_change = abs(largest_norm_difference - previous_difference)
        previous_difference = largest_norm_difference
        centroids = centroids_new
        print(difference_change)
        if difference_change < 0.000001:
            break
        elif np.isnan(difference_change):
            break

    centroids = label_clusters(data, centroids, n)
    return data, centroids


# Read file
# file_path = 'digit-recognizer/train.csv'

# X_train, y_train_labels = loadlocal_mnist(
#                     images_path='digit-recognizer/train-images-idx3-ubyte',
#                     labels_path='digit-recognizer/train-labels-idx1-ubyte')

# X_test, y_test_labels = loadlocal_mnist(
#                 images_path='digit-recognizer/t10k-images-idx3-ubyte',
#                 labels_path='digit-recognizer/t10k-labels-idx1-ubyte')

# train = np.insert(X_train, 0, y_train_labels, axis=1)
# train = np.insert(train, 0, np.zeros(train.shape[0]), axis=1)
# train = train[:500, :]

# test = np.insert(X_test, 0, y_test_labels, axis=1)
# test = np.insert(test, 0, np.zeros(test.shape[0]), axis=1)
# test = test[:500, :]

# data = pd.read_csv(file_path)
# test = data.values
# test = np.insert(test, 0, np.zeros(test.shape[0]), axis=1)
# train = test[:40000, :]
# to_test = test[40000:, :]

A = np.random.normal(1, 0.5, (100, 2))
A = np.insert(A, 0, np.zeros(A.shape[0]) + 1, axis=1)
B = np.random.normal(2, 0.2, (100, 2))
B = np.insert(B, 0, np.zeros(B.shape[0]) + 2, axis=1)
C = np.random.normal(1.5, 0.1, (100, 2))
C = np.insert(C, 0, np.zeros(B.shape[0]) + 15, axis=1)
test = np.vstack((A, B))
test = np.vstack((test, C))
test = np.insert(test, 0, np.zeros(test.shape[0]), axis=1)
my_point = np.array([[-1, -1, 1, 2], [-1, -1, 1, 1], [-1, -1, 2, 2]])

# print(test)

test, cent = K_means(test, 3)
# print(cent)
# start = time.time()
classified = classify(my_point, cent)
# end = time.time()
# print(end - start)
print(get_accuracy(classified[:, 0], classified[:, 1]))

plt.scatter(test[:, 2], test[:, 3], c=test[:, 0])
plt.scatter(my_point[:, 2], my_point[:, 3], marker='d', s=100, label='my point')
plt.scatter(cent[0, 2], cent[0, 3], marker='x', label=str(cent[0,0]))
plt.scatter(cent[1, 2], cent[1, 3], marker='x', label=str(cent[1,0]))
plt.scatter(cent[2, 2], cent[2, 3], marker='x', label=str(cent[2,0]))
# plt.scatter(cent[3, 2], cent[3, 3], marker='x', label=str(cent[3,0]))
# plt.scatter(cent[4, 2], cent[4, 3], marker='x', label=str(cent[4,0]))
plt.legend()
plt.show()



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


