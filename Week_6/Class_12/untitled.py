import numpy as np
import xgboost as xgb

data = np.random.rand(5, 3)
label = np.random.randint(2, size=5)

dtrain = xgb.DMatrix(data, label=label)

print(data)
print(label.shape)
print(dtrain)