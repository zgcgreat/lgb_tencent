# -*- encoding:utf-8 -*-

"""
合并文件，构造训练集和测试集
user_app_actions.csv和user_installedapps.csv暂时没用
"""

import pandas as pd

input_path = '../data_ori'
output_path = '../data'

tr = '{0}/train.csv'.format(output_path)
te = '{0}/test.csv'.format(output_path)

train = '{0}/train.csv'.format(input_path)
test = '{0}/test.csv'.format(input_path)
user = '{0}/user.csv'.format(input_path)
user_app_action = '{0}/user_app_actions.csv'.format(input_path)
user_installedapp = '{0}/user_installedapps.csv'.format(input_path)
app_categorie = '{0}/app_categories.csv'.format(input_path)
position = '{0}/position.csv'.format(input_path)
ad = '{0}/ad.csv'.format(input_path)


def time_transform(series):
    return str(series[2:4])


def convert_age(age):
    return int(int(age) / 5)


def make_data(data, users, positions, ads, app_categories):
    data = pd.merge(data, users, how='left', on='userID')
    data = pd.merge(data, positions, how='left', on='positionID')
    data = pd.merge(data, ads, how='left', on='creativeID')
    data = pd.merge(data, app_categories, how='left', on='appID')
    data['age'] = data['age'].apply(convert_age)
    return data


if __name__ == '__main__':
    users = pd.read_csv(user)
    positions = pd.read_csv(position)
    ads = pd.read_csv(ad)
    app_categories = pd.read_csv(app_categorie)

    train = pd.read_csv(train, dtype={'clickTime': object})
    train = make_data(train, users, positions, ads, app_categories)
    del train['conversionTime']
    train.to_csv(tr, index=False)
    del train
    print('train data completed !!!')

    test = pd.read_csv(test, dtype={'clickTime': object})
    test = make_data(test, users, positions, ads, app_categories)
    test.to_csv(te, index=False)
    del test
    print('test data completed !!!')
