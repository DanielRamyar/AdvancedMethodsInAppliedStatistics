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

kmeans = KMeans(n_clusters=nr_unique_digits, random_state=0).fit(X_train)
predict_kmeans = kmeans.predict(X_test)
print(predict_kmeans)

# plt.imshow(X_train[3, :].reshape((28, 28)))
# plt.figure()
# plt.imshow((X_train[9, :].reshape((28, 28))))
# plt.show()

