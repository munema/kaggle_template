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


class OverallQual_target(Feature):
    def create_features(self):
        column = "OverallQual"

        df_target_train = pd.DataFrame(index=train_origin.index, columns=["{}".format(column) + "_target"])
        for train_index, valid_index in kf.split(train_origin):
            X_tr = train_origin.iloc[train_index, :]
            X_va = train_origin.iloc[valid_index, :]
            y_tr = Y_train_origin[train_index]
            y_va = Y_train_origin[valid_index]

            df_te = data.target_encoding(X_tr, X_va, column, y_va)
            df_target_train.iloc[train_index, :] = df_te.values
        df_target_test = data.target_encoding(test_origin, train_origin, column, Y_train_origin)

        data.check_columns_size(df_target_train,df_target_test)
        self.train = df_target_train
        self.test = df_target_test

class YearBuilt_target(Feature):
    def create_features(self):
        all_bin = data.binning(df_all_origin,"YearBuilt",10)
        train_bin = all_bin[:train_test_split_index]
        test_bin = all_bin[train_test_split_index:].reset_index(drop=True)
        column = train_bin.columns[0]

        df_target_train = pd.DataFrame(index=train_bin.index, columns=["{}".format(column) + "_target"])
        for train_index, valid_index in kf.split(train_bin):
            X_tr = train_bin.iloc[train_index, :]
            X_va = train_bin.iloc[valid_index, :]
            y_tr = Y_train_origin[train_index]
            y_va = Y_train_origin[valid_index]
            df_te = data.target_encoding(X_tr, X_va, column, y_va)
            df_target_train.iloc[train_index, :] = df_te.values
        df_target_test = data.target_encoding(test_bin, train_bin, column, Y_train_origin)

        data.check_columns_size(df_target_train,df_target_test)
        self.train = df_target_train
        self.test = df_target_test


#カテゴリ変数を順序を考慮してorder encoding
class Qual_order_encoding(Feature):
    def create_features(self):
        columns_list = ["KitchenQual","GarageQual","ExterQual","BsmtQual"]
        dict_list =[{"Ex" : 3,"Gd" : 2, "TA" : 1, "Fa" : -1, "Po" : -2, np.nan :-3},
                    {"Ex": 3, "Gd": 2, "TA": 1, "Fa": -1, "Po": -2,np.nan : -3},
                    {"Ex": 3, "Gd": 2, "TA": 1, "Fa": -1, "Po": -2, np.nan :-3},
                    {"Ex": 3, "Gd": 2, "TA": 1, "Fa": -1, "Po": -2, np.nan: -3}]
        df_new_train = pd.DataFrame(index=X_train_origin.index)
        df_new_test = pd.DataFrame(index=test_origin.index)

        for i in range(len(columns_list)):
            df_new_train = pd.concat([df_new_train,data.ordinal_ordered_encoding(X_train_origin, columns_list[i], dict_list[i])],axis=1)

        for i in range(len(columns_list)):
            df_new_test = pd.concat([df_new_test,data.ordinal_ordered_encoding(test_origin, columns_list[i], dict_list[i])],axis=1)

        data.check_columns_size(df_new_train, df_new_test)
        self.train = df_new_train
        self.test = df_new_test


class   Garage__area_order_mix(Feature):
    def create_features(self):
        column = "GarageQual"
        dict ={"Ex" : 5,"Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1, np.nan :0}
        df_new_train = pd.DataFrame(index=X_train_origin.index)
        df_new_test = pd.DataFrame(index=test_origin.index)
        df_new_train = pd.concat([df_new_train,data.ordinal_ordered_encoding(X_train_origin, column, dict)],axis=1)
        df_new_test = pd.concat([df_new_test, data.ordinal_ordered_encoding(test_origin, column, dict)], axis=1)

        tr = pd.DataFrame(np.ravel(df_new_train.values) * X_train["GarageArea"], index=X_train_origin.index)
        tr = tr.rename(columns={tr.columns[0]: "{}".format(column) + "_area_order"})


        te = pd.DataFrame(np.ravel(df_new_test.values)*test["GarageArea"],index=test_origin.index)
        te = te.rename(columns={te.columns[0]: "{}".format(column) + "_area_order"})

        self.train = tr
        self.test = te


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

