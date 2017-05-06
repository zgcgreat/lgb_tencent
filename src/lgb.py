# coding: utf-8

import lightgbm as lgb
import pandas as pd
import scipy as sp
from numpy import *
from sklearn.metrics import roc_auc_score


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


# load or create your dataset
print('Load data...')
df_train = pd.read_csv('../data/validation/train.csv')
df_test = pd.read_csv('../data/validation/test.csv')

y_train = df_train['label'].values
y_test = df_test['label'].values

X_train = df_train.drop('label', axis=1).values
X_test = df_test.drop('label', axis=1).values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'binary_logloss', 'auc'},
    'metric_freq': 1,
    'is_training_metric': 'false',
    'max_bin': 255,
    'num_leaves': 100,
    'learning_rate': 0.1,
    'tree_learner': 'serial',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 100,
    'min_sum_hessian_in_leaf': 100,
    'max_depth': 20
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval
print('The auc of prediction is:', roc_auc_score(y_test, y_pred))
print('The logloss of prediction is:', logloss(y_test, y_pred))

# result to file
fo = open('../result/submission.csv', 'w')
fo.write('instanceID,prob\n')
for t, prob in enumerate(y_pred, start=1):
    fo.write(str(t) + ',' + str(prob) + '\n')
fo.close()




