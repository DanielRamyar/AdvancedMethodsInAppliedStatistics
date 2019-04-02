import numpy as np

A = np.array([[1., 2, 3],
              [5, 4, 2],
              [11, 8, 9],
              [4, 7, 1]])
A = np.insert(A, 0, np.ones(A.shape[0]), axis=1)
A = np.insert(A, 0, np.zeros(A.shape[0]), axis=1)


index = np.random.randint(len(A), size=(2))
init_cent = A[index, :]


print(init_cent)



def dist_centroids(data, cent):
    '''
    Returns the distance between each point (each row in data) and centroids
    '''
    dist = []
    for i in range(cent.shape[0]):
        print(cent[i, 1::], 'lknl')
        temp = np.linalg.norm(data[:, 2::] - cent[i, 2::], axis=1)
        dist = np.append(dist, temp)
    print(dist.shape)
    dist = dist.reshape((cent.shape[0], data.shape[0])).T
    return dist

def label_centroids(data, dist):
    '''
    Adds Label to data according to its closest centroid
    '''

    temp = np.argmin(dist, axis=1) + 1

    data[:, 0] = temp

    return data


disto = dist_centroids(A, init_cent)
print(A)
A = label_centroids(A, disto)
print(A)