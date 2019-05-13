# -- coding: utf-8 --
import argparse
import json
import inspect
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
import feather
import sys
import os
import pandas as pd
import numpy as np
import category_encoders as ce
import seaborn as sns
from sklearn.ensemble import IsolationForest
sys.path.append(os.getcwd())


#Input Dataframe取得
def get_origin_dataframe():
    train = feather.read_dataframe('./data/input/train.feather')
    test = feather.read_dataframe('./data/input/test.feather')
    return train, test

#前処理済みDataframe取得
def get_preprocessing_dataframe():
    X_train = feather.read_dataframe('./data/input/finish_preprocessing_X_train.feather')
    Y_train = feather.read_dataframe('./data/input/finish_preprocessing_Y_train.feather')
    X_test = feather.read_dataframe('./data/input/finish_preprocessing_X_test.feather')
    return X_train, Y_train, X_test

#ベースラインDataframe取得
def get_base_line_dataframe():
    X_train = feather.read_dataframe('./data/input/base_line_train.feather')
    X_test = feather.read_dataframe('./data/input/base_line_test.feather')
    return X_train, X_test

#生成した特徴量のDataframe取得
def get_create_features_dataframe():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    features_list = config["features"]
    df_train = pd.DataFrame(index=[], columns=[])
    df_test = pd.DataFrame(index=[], columns=[])
    for feature in features_list:
        df_feat_train = feather.read_dataframe('./features/feather/{}_train.feather'.format(feature))
        df_feat_test = feather.read_dataframe('./features/feather/{}_test.feather'.format(feature))
        df_train = pd.concat([df_train,df_feat_train],axis=1)
        df_test = pd.concat([df_test,df_feat_test],axis=1)
    return df_train, df_test


def get_dataframe_set(df_train, df_test):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    # 予測に使わないIDを保持・データフレームから削除
    ID = config["ID_name"]
    TARGET = config["target_name"]
    df_train.drop([ID], axis=1, inplace=True)
    df_test.drop([ID], axis=1, inplace=True)

    # 目的変数を別に取っておく
    y_train = df_train[TARGET].values
    df_train.drop([TARGET], axis=1, inplace=True)

    # 学習用データとテストデータの境目のインデックスを保持
    train_test_split_index = len(df_train)

    # 学習用データとテストデータを一度統合する
    df_all = pd.concat((df_train, df_test), ignore_index=True)

    return df_all, y_train, train_test_split_index

def check_columns_size(train, test):
    if  train.shape[1] != test.shape[1]:
        print("-------Warning ! : columns size not same :{} and {}-------".format(train.shape[1],test.shape[1]))

#カテゴリ変数のカラムリスト取得
def get_category_columns(df):
    categoricals = df.dtypes[df.dtypes == "object"].index.values.tolist()
    return categoricals

#数値変数のカラムリスト取得
def get_number_columns(df):
    numbers = df.dtypes[df.dtypes != "object"].index.values.tolist()
    return numbers

#one hot encoding (カテゴリ)
def one_hot_encoding(df):
    ce_ohe = ce.OneHotEncoder(cols=df.columns.tolist(), use_cat_names=True, drop_invariant=True)
    df_onehot = ce_ohe.fit_transform(df)
    return df_onehot

def one_hot_encoding_train_test(train, test):
    ce_ohe = ce.OneHotEncoder(cols=train.columns.tolist(), use_cat_names=True, drop_invariant=True)
    ce_ohe.fit(train)
    train_one_hot = ce_ohe.transform(train)
    test_one_hot = ce_ohe.transform(test)
    print(test_one_hot.shape[1])
    print(train_one_hot.shape[1])
    print(test_one_hot.shape[1]>train_one_hot.shape[1])
    if  train_one_hot.shape[1] < test_one_hot.shape[1]:
        train_one_hot = ce_ohe.transform(train)
        test_one_hot = ce_ohe.transform(test)
    return train_one_hot, test_one_hot

#ordinal encoding (カテゴリ)
def ordinal_encoding(df):
    ce_ord = ce.OrdinalEncoder(cols=df.columns.tolist())
    df_ord = ce_ord.fit_transform(df)
    return df_ord


#ordinal encoding (カテゴリの順序を考慮して ordinal encoding)
#dict : カテゴリと順序
def ordinal_ordered_encoding(df, column, dict):
    df_map = pd.DataFrame(df[column].map(dict), index=df.index.values)
    df_map = df_map.rename(columns={df_map.columns[0] : "{}".format(column)+"_ordered"})
    return df_map


#ターゲットエンコーディング
def target_encoding(df_train, df_valid, column, target):
    target = pd.DataFrame(target, columns=["target"])

    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    target.reset_index(drop=True, inplace=True)
    df_valid = pd.concat([df_valid[column], target], axis=1)
    df_target = df_valid.groupby([column]).sum()
    se_target = pd.Series(df_target["target"], index=df_target.index)

    for c in df_train[column].value_counts().index:
        if c not in df_valid[column].value_counts().index:
            se_target[c] = 0
    se_target = se_target / len(se_target)
    dict = se_target.to_dict()
    df = pd.DataFrame(df_train[column].replace(dict))
    df.columns = ["{}_target".format(column)]

    return df

#カウントエンコーディング
def count_encoding(df, column):
    dict = df[column].value_counts().to_dict()
    count = df[column].replace(dict)
    count = pd.DataFrame(count)
    count.columns=["{}_count".format(column)]
    return count


#ランクカウントエンコーディング
def rank_count_encoding(df, columns_list):
    df_new = pd.DataFrame(index=df.index, columns=[])
    for column in columns_list:
        count_rank = df.groupby(column).transform("count").rank(ascending=False)
        df_new['{}_count_rank'.format(column)] = df[column].map(count_rank)
    return df_new

#単位円に変換(周期)
def circle_encoding(df, columns_list):
    df_new = pd.DataFrame(index=df.index, columns=[])
    for column in columns_list:
        max = df[column].max()
        df_new["{}_sin".format(column)] = df[column].apply(lambda x: np.sin((x / max) * 2 * round(np.pi, 2)))
        df_new["{}_cos".format(column)] = df[column].apply(lambda x: np.cos((x / max) * 2 * round(np.pi, 2)))
    return df_new

def binning(df, column ,num=8):
    df_new = pd.DataFrame(index=df.index, columns=[])
    df_new["{}_bin".format(column)] = pd.cut(df[column], num, labels=False)
    return df_new


#dfのカテゴリ変数をone hot encoding
def category_columns_to_one_hot(df):
    category_list = get_category_columns(df)
    df_one_hot = one_hot_encoding(df[category_list])
    return df_one_hot

#dfのカテゴリ変数をone hot encoding
def category_columns_to_one_hot_train_test(train, test):
    category_list = get_category_columns(train)
    df_one_hot_train, df_one_hot_test = one_hot_encoding_train_test(train[category_list], test[category_list])
    return df_one_hot_train, df_one_hot_test

#数値カラムにNANフラグを立てる
def get_nan_flag(df_train, df_test):
    #欠損値を含む数値カラム
    df_new = pd.DataFrame(index=[], columns=[])
    null_number_list = get_number_columns(df_train[df_train.isnull().any()[df_train.isnull().any() == True].index])
    split_index = len(df_train)
    df = pd.concat([df_train, df_test])
    for column in null_number_list:
        df_new["{}_is_nan".format(column)] = df[column].apply(lambda x : 1 if np.isnan(x) else 0)
    df_train = df_new[:split_index]
    df_test = df_new[split_index:].reset_index(drop=True)
    return df_train, df_test

#ゼロフラグを立てる
def get_zero_flag(df, percent):
    df_new = pd.DataFrame(index=df.index, columns=[])
    number_list = get_number_columns(df)
    sum = len(df)
    for column in number_list:
        if ((df[column]==0).sum() / sum) > percent:
            df_new["{}_is_zero".format(column)] = df[column].apply(lambda x: 1 if x == 0 else 0)
    return df_new

#ゼロの割合
def get_zero_percent(df):
    df_new = pd.DataFrame(index=df.index, columns=[])
    number_list = get_number_columns(df)
    sum = len(df)
    for column in number_list:
        zero_per = (df[column]==0).sum() / sum
        if  zero_per != 0 and zero_per != 1:
            df_new["{}_is_zero".format(column)] = df[column].apply(lambda  x: zero_per if x == 0 else 0)
    return df_new

#none→np.nan
def none_to_nan(df):
    for column in df.columns.values:
        df[column] = df[column].apply(lambda x : np.nan if x == None or x=="None" or x=="NONE" or x=="none" or x=="Nan" or x=="nan" or x=="NAN" else x)

#外れ値除去
def outliers_delete(X_train,y_train, percent=0.05):
    clf = IsolationForest(random_state=3, n_estimators=500, contamination=percent, behaviour='new')
    clf.fit(X_train)
    mv_outliers = pd.DataFrame(clf.predict(X_train))
    mv_outliers.columns = ['OutlierFlag']
    print("outlier counts : {}".format(mv_outliers[mv_outliers == -1].count()))
    y = pd.DataFrame(y_train)
    y.columns = ['Target']
    df_all_olcheck = pd.concat([mv_outliers, X_train, y], axis=1)
    df_all_r = df_all_olcheck[df_all_olcheck['OutlierFlag'] == 1].copy()
    df_all_r.drop("OutlierFlag", axis=1, inplace=True)
    y_train_r = df_all_r["Target"].values
    df_all_r.drop(["Target"], axis=1, inplace=True)
    return df_all_r, y_train_r

#相関の高いカラム取得
def high_corr(X_train, percent):
    list = []
    #共分散行列
    corr = X_train.corr()
    #上三角行列
    tri = pd.DataFrame(np.triu(corr, k=1), index=corr.index, columns=corr.index)

    for column in corr.columns:
        for index in corr.index:
            if tri[column][index] > percent or tri[column][index] < -percent:
                list.append((column, index))
    return list

#ターゲットと相関の高いDataframe取得
def high_corr_target(train):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    #共分散行列
    corr = train.corr()
    #ターゲットとの相関
    c = corr[config["target_name"]].apply(lambda x: np.fabs(x)).sort_values(ascending=False)
    c = c.drop([config["target_name"]])
    df_c = pd.DataFrame(c.values, index=c.index, columns=["corr_to_target"])
    return df_c