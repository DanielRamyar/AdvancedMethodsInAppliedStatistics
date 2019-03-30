import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# Read file
file_path = 'digit-recognizer/train.csv'
data = pd.read_csv(file_path, nrows=2000)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

nr_unique_digits = len(np.unique(y_train))

kmeans = KMeans(n_clusters=50, random_state=0).fit(X_train)
predict_kmeans = kmeans.predict(X_test)

print(kmeans.labels_[0:20])
# for i in range(10):
    # plt.figure()
    # plt.imshow(X_train[i, :].reshape((28, 28)))
    
plt.figure()
plt.imshow(X_train[5, :].reshape((28, 28)))
plt.figure()
plt.imshow(X_train[14, :].reshape((28, 28)))
plt.show()

