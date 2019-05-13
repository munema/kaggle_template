# -- coding: utf-8 --
import pandas as pd
import feather
import numpy as np
import re as re
import argparse
import json
import sys
import os
from sklearn.externals import joblib
sys.path.append(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
from base import Feature, get_arguments, generate_features, generate_dataframe, get_features_to_json
from function import data, config_json


args = get_arguments()
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

#元データ
train_origin, test_origin = data.get_origin_dataframe()

#ターゲットとの相関
target_corr = data.high_corr_target(train_origin)

# IDとを削除・Targetをy_trainとして分離
df_all, y_train, train_test_split_index = data.get_dataframe_set(train_origin, test_origin)

#none → nan
data.none_to_nan(df_all)

print("befor columns {}".format(df_all.shape[1]))

#カテゴリ変数を削除
df_all= df_all.drop(data.get_category_columns(df_all), axis=1)

print("after columns {}".format(df_all.shape[1]))

print("befor nan pacent {}".format(df_all.isnull().sum().sum()/df_all.size))

#数値変数の欠損率を表示
print("---detail---")
print("categorical values delete and nan values delete")
print(df_all[df_all.isnull().any()[df_all.isnull().any()==True].index].apply(lambda x: (x.isnull().sum())/len(x)))
print("------------")

#欠損値を中央値で埋める
df_all = df_all.fillna(df_all.mode().iloc[0])

print("after nan pacent {}".format(df_all.isnull().sum().sum()/df_all.size))

train = df_all[:train_test_split_index]

df_y_train = pd.DataFrame(y_train, columns = [config["target_name"]]).reset_index(drop=True)

test = df_all[train_test_split_index:]
test = test.reset_index(drop=True)

if __name__ == '__main__':
    train.to_feather('./data/input/' + "finish_preprocessing_X_train" + '.feather')
    df_y_train.to_feather('./data/input/' + "finish_preprocessing_Y_train" + '.feather')
    test.to_feather('./data/input/' + "finish_preprocessing_X_test" + '.feather')
    joblib.dump(target_corr, "./data/input/target_and_train_origin_corr", compress=True)