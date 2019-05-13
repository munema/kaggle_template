# -- coding: utf-8 --
import pandas as pd
import feather
import numpy as np
import re as re
import argparse
from sklearn.model_selection import KFold,StratifiedKFold
import json
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
from base import Feature, get_arguments, generate_features, generate_dataframe, get_features_to_json
from function import data, config_json

Feature.dir = 'features'

class one_hot_encoding(Feature):
    def create_features(self):
        train_oht, test_oht = data.category_columns_to_one_hot_train_test(X_train_origin, test_origin)

        data.check_columns_size(train_oht, test_oht)
        self.train = train_oht
        self.test = test_oht

#数値カラムのNanフラグ
class Nan_flag(Feature):
    def create_features(self):
        train_nan, test_nan = data.get_nan_flag(X_train_origin, test_origin)

        data.check_columns_size(train_nan, test_nan)
        self.train = train_nan
        self.test = test_nan

#数値カラムの0カウントエンコーディング
class Zero_percent(Feature):
    def create_features(self):
        df_all_zero_per = data.get_zero_percent(df_all_origin)
        train_zero_per = df_all_zero_per[:train_test_split_index]
        test_zero_per = df_all_zero_per[train_test_split_index:]
        test_zero_per = test_zero_per.reset_index(drop=True)

        data.check_columns_size(train_zero_per, test_zero_per)
        self.train = train_zero_per
        self.test = test_zero_per

if __name__ == '__main__':
    args = get_arguments()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    cv = config["cv"]
    if cv == "kfold":
        kf = KFold(**config["kfold"])
    elif cv == "skfold":
        kf = StratifiedKFold(**config["skfold"])

    #元データ
    train_origin, test_origin = data.get_origin_dataframe()

    #none→nan
    data.none_to_nan(train_origin)
    data.none_to_nan(test_origin)

    #df_all_origin : train_origin(-target) + test_origin
    df_all_origin, Y_train_origin, train_test_split_index = data.get_dataframe_set(train_origin, test_origin)
    #X_train_origin : train_origin(-target)
    X_train_origin = df_all_origin[:train_test_split_index]


    #前処理済みデータ
    X_train, Y_train, test =data.get_preprocessing_dataframe()

    #X_train + Y_train
    train_all = pd.concat([X_train, Y_train], axis=1)

    #X_train + test
    df_all = pd.concat([X_train, test],sort=False, ignore_index=True)


    #特徴量作成
    generate_features(globals(), args.force)

    #作成した特徴量でDataFrame作成
    df_train, df_test = generate_dataframe(globals())

    #作成した特徴量をconfigに追加
    get_features_to_json(globals(),True)

