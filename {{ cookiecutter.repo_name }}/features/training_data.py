# -- coding: utf-8 --
#base lineモデル

import pandas as pd
import feather
import numpy as np
import re as re
import argparse
import json
import sys
import os
sys.path.append(os.getcwd())
from function import data, model_train, submit

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))


#データ取得
#前処理済みデータ
X_train, Y_train, X_test = data.get_preprocessing_dataframe()
y_train = Y_train.values.reshape(-1,)

#生成データ
X_create_train, X_create_test = data.get_create_features_dataframe()

#前処理済みデータと生成データを結合
X_train = pd.concat([X_train, X_create_train], axis=1)
X_test = pd.concat([X_test, X_create_test], axis=1)

#訓練データと正解データも結合
train = pd.concat([X_train, Y_train], axis=1)

print("train shape : {} is_nan : {}".format(train.shape, X_train.isnull().any().any()))
print("X_test shape : {} is_nan : {}".format(X_test.shape, X_test.isnull().any().any()))

if __name__ == '__main__':
    train.to_feather('./data/input/' + "base_line_train" + '.feather')
    X_test.to_feather('./data/input/' + "base_line_test" + '.feather')