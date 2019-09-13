# feature importance
# local score 0.9355670019874779
# kaggle score 0.8967

import os
import sys  # noqa
from time import time
from pprint import pprint  # noqa
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
np.set_printoptions(threshold=sys.maxsize)

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'

train_file = 'ieee-fraud-detection/train_transaction' if is_kaggle else 'ieee-fraud-detection/train_transaction-sample'
test_file = 'ieee-fraud-detection/test_transaction' if is_kaggle else 'ieee-fraud-detection/test_transaction-sample'

train_identity_file = 'ieee-fraud-detection/train_identity' if is_kaggle else 'ieee-fraud-detection/train_identity'
test_identity_file = 'ieee-fraud-detection/test_identity' if is_kaggle else 'ieee-fraud-detection/test_identity'

start_time = time()
last_time = time()


def timer():
    global last_time

    print(f'{((time() - last_time) / 60):.1f}, {((time() - start_time) / 60):.1f} mins\n')

    last_time = time()

# Based on https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65


def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2

    NAlist = []  # Keeps track of columns that have missing values filled in.

    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings

            previous_type = df[col].dtype

            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            # print(f"{col} before: {previous_type}, after: {df[col].dtype}")

    mem_usg = df.memory_usage().sum() / 1024**2

    print(f"Memory usage is: {mem_usg}MB, {100 * mem_usg / start_mem_usg:.1f}% of the initial size")

    return df, NAlist


def evaluate(train, test, unique_id, target):

    print('evaluate')

    # binary
    # 'metric': 'auc',
    lgb_model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1, objective='binary', metric='auc')

    x_train = train.drop([target, unique_id], axis=1)
    y_train = train[target]

    x_test = test[x_train.columns]

    lgb_model.fit(x_train, y_train)

    train_predictions = lgb_model.predict(x_train)
    test_predictions = lgb_model.predict(x_test)

    train_score = roc_auc_score(y_train, train_predictions)

    timer()

    return test_predictions, train_score

# --- missing values


def get_many_missing_values(train, test, unique_id, target):

    print(f'get_many_missing_values')

    train_targets = train[target]

    threshold = 0.75

    train_missing = (train.isnull().sum() / len(train)).sort_values(ascending=False)
    test_missing = (test.isnull().sum() / len(test)).sort_values(ascending=False)

    # identify missing values above threshold
    train_missing = train_missing.index[train_missing > threshold]
    test_missing = test_missing.index[test_missing > threshold]

    all_missing = list(set(set(train_missing) | set(test_missing)))

    if len(all_missing) > 0:
        print(f'columns with more than {100 * threshold}% missing values')
        pprint(all_missing)

        train = train.drop(columns=all_missing, axis=1)
        test = test.drop(columns=all_missing, axis=1)

        train, test = train.align(test, join='inner', axis=1)

        # restore after align
        train[target] = train_targets

    timer()

    return train, test

# --- remove keys


def remove_keys(list, keys):

    result = [x for x in list if x not in keys]

    return result


# --- test if column is numeric

def is_numeric(col):
    col_type = str(col.dtype)

    return col_type.startswith('int') | col_type.startswith('uint') | col_type.startswith('float')

# --- replace missing values


def replace_missing_values(train, test, unique_id, target):

    print(f'replace_missing_values')

    numeric_cols = [col for col in train.columns
                    if is_numeric(train[col])]

    numeric_cols = remove_keys(numeric_cols, [unique_id, target])

    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
    categorical_cols = remove_keys(categorical_cols, [unique_id, target])

    # replace missing numericals with mean
    for col in numeric_cols:
        if train[col].isna().any() | test[col].isna().any():
            mean = train[col].mean()

            train[col].fillna(mean, inplace=True)

            if col in test.columns:
                test[col].fillna(mean, inplace=True)

    # convert to lowercase
    for col in categorical_cols:
        train[col] = train[col].apply(lambda x: str(x).lower())

        if col in test.columns:
            test[col] = test[col].apply(lambda x: str(x).lower())

    # replace string nan with np.nan
    train.replace('nan', np.nan, inplace=True)
    test.replace('nan', np.nan, inplace=True)

    # replace missing categoricals with mode
    for col in categorical_cols:
        if train[col].isna().any() or test[col].isna().any():
            mode = train[col].mode()[0]

            train[col].fillna(mode, inplace=True)

            if col in test.columns:
                test[col].fillna(mode, inplace=True)

    timer()

    return train, test

# --- column differences


def get_column_differences(train, test, unique_id, target):

    print(f'get_column_differences')

    train_without_target = train.drop(target, axis=1)

    not_in_test = train_without_target.columns.difference(test.columns)
    not_in_train = test.columns.difference(train_without_target.columns)

    if len(not_in_test) > 0:
        print(f'In train but not test\n{not_in_test}')

    if len(not_in_train) > 0:
        print(f'In test but not train\n{not_in_train}')

    timer()

    return train, test


# --- categorical data


def get_categorical_data(train, test, unique_id, target):

    print(f'get_categorical_data')

    train_targets = train[target]

    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']

    if unique_id in categorical_cols:
        categorical_cols.remove(unique_id)

    max_categories = train.shape[0] * 0.5

    too_many_value_categorical_cols = [col for col in categorical_cols
                                       if train[col].nunique() >= max_categories]

    if len(too_many_value_categorical_cols) > 0:
        print('too many categorical values', too_many_value_categorical_cols)

    # drop if too many values - usually a unique id column

    categorical_cols = [i for i in categorical_cols if i not in too_many_value_categorical_cols]

    train = train.drop(too_many_value_categorical_cols, axis=1)
    test.drop([col for col in too_many_value_categorical_cols
               if col in test.columns], axis=1, inplace=True)

    # one-hot encode if not too many values

    max_ohe_categories = 10

    ohe_categorical_cols = [col for col in categorical_cols
                            if train[col].nunique() <= max_ohe_categories]

    categorical_cols = [i for i in categorical_cols if i not in ohe_categorical_cols]

    if len(ohe_categorical_cols) > 0:
        print('one-hot encode', ohe_categorical_cols)

        # one-hot encode & align to have same columns
        train = pd.get_dummies(train, columns=ohe_categorical_cols)
        test = pd.get_dummies(test, columns=ohe_categorical_cols)
        train, test = train.align(test, join='inner', axis=1)

        # restore after align
        train[target] = train_targets

    # possibly rank encode rather than ohe. see gstore.

    # label encode (convert to integer)

    label_encode_categorical_cols = categorical_cols

    print('label encode', label_encode_categorical_cols)

    for col in label_encode_categorical_cols:
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        test[col] = lbl.transform(list(test[col].values.astype('str')))

    timer()

    return train, test

# --- feature selection


def get_feature_selection(train, test, unique_id, target):

    print(f'get_feature_selection')

    all_numeric_cols = [col for col in train.columns
                        if is_numeric(train[col])]

    if unique_id in all_numeric_cols:
        all_numeric_cols.remove(unique_id)

    if target in all_numeric_cols:
        all_numeric_cols.remove(target)

    # feature selection via variance
    train_numeric = train[all_numeric_cols].fillna(0)
    select_features = VarianceThreshold(threshold=0.2)
    select_features.fit(train_numeric)
    numeric_cols = train_numeric.columns[select_features.get_support(indices=True)].tolist()

    # remove cols without variance
    for col in all_numeric_cols:
        if col not in numeric_cols:
            print(f'variance drop {col}')
            train.drop(col, axis=1, inplace=True)

            if col in test.columns:
                test.drop(col, axis=1, inplace=True)

    timer()

    return train, test

# --- feature importance


def get_feature_importance(train, test, unique_id, target, numeric_only):

    print(f'get_feature_importance')

    model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1)

    x_train = train.drop([unique_id, target], axis=1)

    if numeric_only:
        non_numeric_cols = [col for col in x_train.columns if not is_numeric(train[col])]

        x_train = x_train.drop(non_numeric_cols, axis=1)

    # initialize an empty array to hold feature importances
    feature_importances = np.zeros(x_train.shape[1])

    # fit the model twice to avoid overfitting
    for i in range(2):

        # split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(x_train, train[target],
                                                                            test_size=0.25, random_state=i)

        # train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100,
                  eval_set=[(valid_features, valid_y)],
                  eval_metric='rmse', verbose=False)

        # record the feature importances
        feature_importances += model.feature_importances_

    # average feature importances!
    feature_importances = feature_importances / 2

    feature_importances = pd.DataFrame(
        {'feature': list(x_train.columns), 'importance': feature_importances}).sort_values('importance', ascending=False)

    # sort features according to importance
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index()

    most_important_features = feature_importances[0:10]['feature'].tolist()

    # normalize the feature importances to add up to one
    feature_importances['importance_normalized'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['importance_normalized'])

    # find the features with minimal importance
    # unimportant_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

    # Threshold for cumulative importance
    threshold = 0.9996

    # extract the features to drop

    features_to_drop = list(feature_importances[feature_importances[
        'cumulative_importance'] > threshold]['feature'])

    if len(features_to_drop) > 0:
        print(feature_importances)

        print(f'features to drop, under {threshold} importance:')
        pprint(features_to_drop)

        train = train.drop(features_to_drop, axis=1)
        test = test.drop(features_to_drop, axis=1)

    timer()

    return train, test, most_important_features

# --- remove collinear features


def get_collinear_features(train, test, unique_id, target):

    print('get_collinear_features')

    corrs = train.corr()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(np.bool))

    threshold = 0.8

    # select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if len(to_drop) > 0:
        print('collinear drop')
        pprint(to_drop)

        train = train.drop(columns=to_drop, axis=1)
        test = test.drop(columns=to_drop, axis=1)

    timer()

    return train, test


# statistical features


def get_statistical_features(train, test, unique_id, target):

    print('get_statistical_features')

    numeric_cols = [col for col in train.columns
                    if is_numeric(train[col])]

    numeric_cols = remove_keys(numeric_cols, [unique_id, target])

    for df in [train, test]:
        df['sum'] = df[numeric_cols].sum(axis=1)
        df['min'] = df[numeric_cols].min(axis=1)
        df['max'] = df[numeric_cols].max(axis=1)
        df['mean'] = df[numeric_cols].mean(axis=1)
        df['std'] = df[numeric_cols].std(axis=1)
        df['skew'] = df[numeric_cols].skew(axis=1)
        df['kurt'] = df[numeric_cols].kurtosis(axis=1)
        df['med'] = df[numeric_cols].median(axis=1)
        df['perc_5'] = df[numeric_cols].apply(lambda x: np.percentile(x, 10), axis=1)
        df['perc_10'] = df[numeric_cols].apply(lambda x: np.percentile(x, 10), axis=1)
        df['perc_25'] = df[numeric_cols].apply(lambda x: np.percentile(x, 25), axis=1)
        df['perc_50'] = df[numeric_cols].apply(lambda x: np.percentile(x, 50), axis=1)
        df['perc_75'] = df[numeric_cols].apply(lambda x: np.percentile(x, 75), axis=1)
        df['perc_95'] = df[numeric_cols].apply(lambda x: np.percentile(x, 99), axis=1)
        df['perc_99'] = df[numeric_cols].apply(lambda x: np.percentile(x, 99), axis=1)

    timer()

    return train, test

# arithmetic features


def get_arithmetic_features(train, test, unique_id, target, cols):

    print('get_arithmetic_features')

    numeric_cols = [col for col in cols
                    if is_numeric(train[col])]

    numeric_cols = remove_keys(numeric_cols, [unique_id, target])

    for i1 in range(0, len(numeric_cols)):
        col1 = numeric_cols[i1]

        # powers
        train[f'{col1} squared'] = train[col1] ** 2
        test[f'{col1} squared'] = test[col1] ** 2
        train[f'{col1} cubed'] = train[col1] ** 3
        test[f'{col1} cubed'] = test[col1] ** 3
        train[f'{col1}^4'] = train[col1] ** 4
        test[f'{col1}^4'] = test[col1] ** 4

        for i2 in range(i1 + 1, len(numeric_cols)):
            col2 = numeric_cols[i2]

            train[f'{col1} by {col2}'] = train[col1] * train[col2]
            test[f'{col1} by {col2}'] = test[col1] * test[col2]

            train[f'{col1} plus {col2}'] = train[col1] + train[col2]
            test[f'{col1} plus {col2}'] = test[col1] + test[col2]

            train[f'{col1} minus {col2}'] = train[col1] - train[col2]
            test[f'{col1} minus {col2}'] = test[col1] - test[col2]

            if not (train[col2] == 0).any():
                train[f'{col1} on {col2}'] = train[col1] / train[col2]
                test[f'{col1} on {col2}'] = test[col1] / test[col2]
            elif not (train[col1] == 0).any():
                train[f'{col2} on {col1}'] = train[col2] / train[col1]
                test[f'{col2} on {col1}'] = test[col2] / test[col1]

    timer()

    return train, test


# custom features

def get_custom_features(train, test, unique_id, target):

    print('get_custom_features')
    timer()

    return train, test

# --------------------- run


def run():

    unique_id = 'TransactionID'
    target = 'isFraud'

    # load data

    train = pd.read_csv(f'../input/{train_file}.csv{zipext}')
    test = pd.read_csv(f'../input/{test_file}.csv{zipext}')

    train_identity = pd.read_csv(f'../input/{train_identity_file}.csv{zipext}')
    test_identity = pd.read_csv(f'../input/{test_identity_file}.csv{zipext}')

    # join files on TransactionId

    train = train.set_index(unique_id).join(train_identity.set_index(unique_id)).reset_index()
    test = test.set_index(unique_id).join(test_identity.set_index(unique_id)).reset_index()

    train, _ = reduce_mem_usage(train)
    test, _ = reduce_mem_usage(test)

    train, test = get_many_missing_values(train, test, unique_id, target)

    train, test = replace_missing_values(train, test, unique_id, target)

    train, test = get_column_differences(train, test, unique_id, target)

    train, test = get_statistical_features(train, test, unique_id, target)

    train, test = get_custom_features(train, test, unique_id, target)

    train, test, most_important_cols = get_feature_importance(train, test, unique_id, target, True)

    # use less cols if memory problem
    train, test = get_arithmetic_features(train, test, unique_id, target, most_important_cols[0:5])

    train, test = get_categorical_data(train, test, unique_id, target)

    train, test = get_collinear_features(train, test, unique_id, target)

    train, test = get_feature_selection(train, test, unique_id, target)

    train, test, _ = get_feature_importance(train, test, unique_id, target, False)

    # ----------

    test_predictions, train_score = evaluate(train, test, unique_id, target)

    print('score', train_score)

    test[target] = test_predictions

    predictions = test[[unique_id, target]]

    predictions.to_csv('submission.csv', index=False)


# -------- main

run()

print(f'Finished {((time() - start_time) / 60):.1f} mins\a')
