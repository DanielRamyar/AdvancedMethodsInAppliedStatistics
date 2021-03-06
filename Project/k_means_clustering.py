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
    for i in range(new_cent.shape[0]):
        new = np.average(data[data[:, 0] == i + 1][:, 2::], axis=0)
        new_cent[i, 2::] = new

    return new_cent


def label_clusters(data, cent, n):
    '''
    Finds most common label in cluster and sets cluster label
    '''
    for i in range(n):
        temp = data[data[:, 0] == i + 1]
        if len(temp) == 0:
            continue

        temp1 = temp[:, 1].astype(int)
        label = np.bincount(temp1).argmax()
        cent[i, 0] = label
    return cent


def classify(to_be_classified, cent):

    dist = []
    classified = np.copy(to_be_classified)
    print(cent)
    for i in range(cent.shape[0]):
        temp = np.linalg.norm(to_be_classified[:, 2::] - cent[i, 2::], axis=1)
        dist = np.append(dist, temp)
    dist = dist.reshape((cent.shape[0], to_be_classified.shape[0])).T
    temp = np.argmin(dist, axis=1)

    classified[:, 0] = cent[temp, 0]

    return classified


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

    for _ in range(20):
        print(1)
        distance_from_centroids = dist_centroids(data, centroids)
        data = label_centroids(data, distance_from_centroids)
        centroids_new = new_centroids(data, centroids)

        difference = centroids[:, 2::] - centroids_new[:, 2::]
        norm_difference = np.linalg.norm(difference, axis=1)
        largest_norm_difference = max(norm_difference)
        difference_change = abs((largest_norm_difference - previous_difference) /
            np.mean([largest_norm_difference, previous_difference])) *100
        previous_difference = largest_norm_difference

        centroids = np.copy(centroids_new)
        print(difference_change)
        



    centroids = label_clusters(data, centroids, n)

    return data, centroids


# Read file

np.random.seed(2)
X_train, y_train_labels = loadlocal_mnist(
                    images_path='digit-recognizer/train-images-idx3-ubyte',
                    labels_path='digit-recognizer/train-labels-idx1-ubyte')

X_test, y_test_labels = loadlocal_mnist(
                images_path='digit-recognizer/t10k-images-idx3-ubyte',
                labels_path='digit-recognizer/t10k-labels-idx1-ubyte')

train = np.insert(X_train, 0, y_train_labels, axis=1)
train = np.insert(train, 0, np.zeros(train.shape[0]), axis=1)
train = train[:4000, :]

test = np.insert(X_test, 0, y_test_labels, axis=1)
test = np.insert(test, 0, np.zeros(test.shape[0]), axis=1)
test = test[:100, :]


train, cent = K_means(train, 10)


start = time.time()
print(test)
classified = classify(test, cent)
print(classified)
end = time.time()


accuracy = get_accuracy(classified[:, 0], classified[:, 1])
print('Your accuracy is: ', accuracy)
print('Your error rate is: ', 1 - accuracy)
print('Time to classify: ', end - start)

display_digit(classified, 1)
plt.show()
