import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

plt.style.use('bmh')


def convert_to_Dmatrix(X, y, features_names1):
    dmatrix = xgb.DMatrix(data=X, label=y, feature_names=features_names)
    return dmatrix, X, y


fnames = ['Exam_2019_Prob4_TrainData.csv',
          'Exam_2019_Prob4_TestData.csv',
          'Exam_2019_Prob4_BlindData.csv']

data_dict = {}
features_names = []
ID_blind = []

for file in fnames:
    temp = pd.read_csv(file)
    if file == 'Exam_2019_Prob4_TrainData.csv':
        features_names = list(temp.iloc[:, 1:-1])
    elif file == 'Exam_2019_Prob4_BlindData.csv':
        ID_blind = temp.iloc[:, 0]
    data_dict[file[:-4]] = temp.iloc[:, 1:].values


dmatrix_train, X_train, y_train = convert_to_Dmatrix(data_dict['Exam_2019_Prob4_TrainData'][:, :-1],
                                                     data_dict['Exam_2019_Prob4_TrainData'][:, -1],
                                                     features_names)

dmatrix_test, X_test, y_test = convert_to_Dmatrix(data_dict['Exam_2019_Prob4_TestData'][:, :-1],
                                                  data_dict['Exam_2019_Prob4_TestData'][:, -1],
                                                  features_names)


X_blind = data_dict['Exam_2019_Prob4_BlindData']
dmatrix_blind = xgb.DMatrix(data=X_blind, 
                            feature_names=features_names)


param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
model = xgb.train(param, dmatrix_train, num_boost_round=12)

y_pred = model.predict(dmatrix_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(dmatrix_test.get_label(), predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

binwidth = 0.03
n_bins = np.arange(min(y_pred[y_test==0]), max(y_pred[y_test==0]) + binwidth, binwidth)
plt.hist(y_pred[y_test==0], bins=n_bins, color='r', histtype='step', label='No-show==0', lw=2)
plt.hist(y_pred[y_test==1], bins=n_bins, color='k', histtype='step', label='No-show==1', lw=2)
plt.xlabel('Decision score')
plt.ylabel('Counts')
plt.legend()


xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

y_pred = model.predict(dmatrix_blind)
predictions = [round(value) for value in y_pred]


ID_blind = np.append(ID_blind.values[:, None], np.array(predictions)[:, None], axis=1)
np.savetxt('ramyar.AMAS_Exam_2019.Problem4.NoShowFalse.txt', ID_blind[ID_blind[:, 1] == 0][:, 0], fmt='%i')
np.savetxt('ramyar.AMAS_Exam_2019.Problem4.NoShowTrue.txt', ID_blind[ID_blind[:, 1] == 1][:, 0], fmt='%i')
