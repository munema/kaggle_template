# -- coding: utf-8 --
import argparse
import json
import feather
from sklearn.externals import joblib
import sys
import os
import pandas as pd
import numpy as np
import category_encoders as ce
sys.path.append(os.getcwd())
from function import data

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))


#相関の高いカラムを削除
def delete_high_corr(train, test, percent):
    imp = joblib.load("./features/importances/base_line")
    high_corr_list = data.high_corr(train, percent)
    print("high corr counts : {}".format(len(high_corr_list)))
    for tuple in high_corr_list:
        if  (imp.ix[tuple[0]] < imp.ix[tuple[1]])[0] and tuple[0] in train.columns:
            train = train.drop(tuple[0], axis=1)
            test = test.drop(tuple[0], axis=1)
            print("delete {}".format(tuple[0])+" : {}".format(imp.ix[tuple[0]]))
        elif    (imp.ix[tuple[0]] >= imp.ix[tuple[1]])[0] and tuple[1] in train.columns:
            train = train.drop(tuple[1], axis=1)
            test = test.drop(tuple[1], axis=1)
            print("delete {}".format(tuple[1]) + " : {}".format(imp.ix[tuple[1]]))
    return train, test


def delete_importance_0(train, test):
    imp = joblib.load("./features/importances/base_line")
    imp_0_index=imp.query("importance == 0").index
    train = train.drop(imp_0_index, axis=1)
    test = test.drop(imp_0_index, axis=1)
    return train, test

#保存
if __name__ == '__main__':
    # データ取得
    train, X_test = data.get_base_line_dataframe()
    y_train = train[config["target_name"]]
    X_train = train.drop([config["target_name"]], axis=1)

    # 外れ値を除去
    X_train, y_train = data.outliers_delete(X_train, y_train, percent=0.01)

    # インポータンス0のカラムを削除
    X_train, X_test = delete_importance_0(X_train, X_test)

    X_train, X_test = delete_high_corr(X_train,X_test, 0.9)