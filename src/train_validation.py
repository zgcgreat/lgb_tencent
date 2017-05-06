# -*- encoding:utf-8 -*-

"""
划分训练集和验证集
把第30天的数据作为验证集
"""
import pandas as pd


def time_transform(series):
    return str(series)[2:4]

df = pd.read_csv('../data/train.csv')
df = df.drop_duplicates()
train = df[df['clickTime'] < 300000]

train['clickTime'] = train['clickTime'].apply(time_transform)

train.to_csv('../data/validation/train.csv', index=False)
print(len(train))

test = df.ix[df.index.difference(train.index)]
del train
test['clickTime'] = test['clickTime'].apply(time_transform)

test.to_csv('../data/validation/test.csv', index=False)
print(len(test))
