# baseline: constant prob 0.5
# kaggle score

import sys  # pylint: disable=unused-import
import numpy as np
import pandas as pd
from time import time

import os

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = '' if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train_transaction.csv{zipext}')
test = pd.read_csv(f'../input/test_transaction.csv{zipext}')

#-------- main

start_time = time()

target = 'isFraud'

result = 0.5

train['isFraud'] = result

test[target] = result


predictions = test[['TransactionID', target]]

predictions.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
