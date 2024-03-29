# baseline: constant 0.5
# kaggle score 0.5

import sys  # noqa
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score

import os

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = '' if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train_transaction.csv{zipext}')
test = pd.read_csv(f'../input/test_transaction.csv{zipext}')

# -------- main

start_time = time()

target = 'isFraud'

prediction = 0

train['predicted'] = prediction

score = roc_auc_score(train[target], train.predicted)
print(score)

test[target] = 0.5

predictions = test[['TransactionID', target]]

predictions.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
