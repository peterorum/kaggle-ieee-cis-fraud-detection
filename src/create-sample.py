# create sample of data

import pandas as pd

sample_size = 10000


train = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv.zip")
sample = train.sample(sample_size)
sample.to_csv('../input/ieee-fraud-detection/train_transaction-sample.csv', index=False)

test = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv.zip")
sample = test.sample(sample_size)
sample.to_csv('../input/ieee-fraud-detection/test_transaction-sample.csv', index=False)
