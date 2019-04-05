import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score

plt.style.use('bmh')
def convert_to_Dmatrix(data_set_1, data_set_2):

    labels_1 = np.zeros((data_set_1.shape[0], 1))
    labels_2 = np.ones((data_set_2.shape[0], 1))

    X = np.vstack((data_set_1, data_set_2))
    y = np.vstack((labels_1, labels_2))

    dmatrix = xgb.DMatrix(data=X, label=y, feature_names=['x', 'y', 'z'])

    return dmatrix, X, y



fnames = ['BDT_background_train.txt', 'BDT_signal_train.txt',
          'BDT_background_test.txt', 'BDT_signal_test.txt']

data_dict = {}

for file in fnames:
    data_dict[file[:-4]] = np.loadtxt(file)

dmatrix_train, X_train, y_train = convert_to_Dmatrix(data_dict['BDT_background_train'],
                                                     data_dict['BDT_signal_train'])
dmatrix_test, X_test, y_test = convert_to_Dmatrix(data_dict['BDT_background_test'],
                                                  data_dict['BDT_signal_test'])

param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
model = xgb.train(param, dmatrix_train)

y_pred = model.predict(dmatrix_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(dmatrix_test.get_label(), predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# plt.style.use('bmh')

# plt.figure(figsize=(15, 8))
# plt.subplot(131)
# plt.hist(data_bg[:, 0], label='Background X', alpha=0.9)
# plt.hist(data_signal[:, 0], label='Signal X', alpha=0.7)
# plt.legend()

# plt.subplot(132)
# plt.hist(data_bg[:, 1], label='Background Y', alpha=0.9)
# plt.hist(data_signal[:, 1], label='Signal Y', alpha=0.7)
# plt.legend()

# plt.subplot(133)
# plt.hist(data_bg[:, 2], label='Background Z', alpha=0.9)
# plt.hist(data_signal[:, 2], label='Signal Z', alpha=0.7)
# plt.legend()

# plt.figure(figsize=(15, 5))
# plt.subplot(131)
# plt.plot(data_bg[:, 0], data_bg[:, 1], '.', label='X vs Y BG')
# plt.plot(data_signal[:, 0], data_signal[:, 1], '.', label='X vs Y Signal')
# plt.legend()

# plt.subplot(132)
# plt.plot(data_bg[:, 0], data_bg[:, 2], '.', label='X vs Z BG')
# plt.plot(data_signal[:, 0], data_signal[:, 2], '.', label='X vs Z Signal')
# plt.legend()

# plt.subplot(133)
# plt.plot(data_bg[:, 1], data_bg[:, 2], '.', label='Y vs Z BG')
# plt.plot(data_signal[:, 1], data_signal[:, 2], '.', label='Y vs Z Signal')
# plt.legend()
# plt.show()


background = np.array([])
signal = np.array([])

for i, prediction in enumerate(predictions):
    if prediction == 0:
        background = np.append(background, X_test[i])
    else:
        signal = np.append(signal, X_test[i])

data_bg = signal.reshape(int(len(signal) / 3), 3)
data_signal = background.reshape(int(len(background) / 3), 3)

plt.figure()
plt.plot(data_bg[:, 0], data_bg[:, 1], 'r.', label='X vs Y BG')
plt.plot(data_signal[:, 0], data_signal[:, 1], 'b.', label='X vs Y Signal')
plt.legend()

data_bg = data_dict['BDT_background_test']
data_signal = data_dict['BDT_signal_test']

plt.figure()
plt.plot(data_bg[:, 0], data_bg[:, 1], 'r.', label='X vs Y BG')
plt.plot(data_signal[:, 0], data_signal[:, 1], 'b.', label='X vs Y Signal')
plt.legend()

plt.show()
