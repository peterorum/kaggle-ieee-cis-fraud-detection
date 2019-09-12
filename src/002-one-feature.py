# amount > $125
# kaggle score 0.5189

import sys  # noqa
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

# add prdict to train and calc overall score

train['predicted'] = train.TransactionAmt.apply(lambda x: 1 if x > 125 else 0)

score = roc_auc_score(train[target], train.predicted)
print(score)

# set same prediction in test target
test[target] = test.TransactionAmt.apply(lambda x: 1 if x > 125 else 0)

# save to submission
predictions = test[['TransactionID', target]]

predictions.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
