# create sample of data

import pandas as pd

train = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv.zip")

sample_size = 10000

sample = train.sample(sample_size)

sample.to_csv('../input/ieee-fraud-detection/train_transaction-sample.csv', index=False)
