# -- coding: utf-8 --
#base line + 外れ値除去＋目的変数を対数化＋インポータンス0のカラムを消去

import pandas as pd
import feather
import numpy as np
import re as re
from sklearn.externals import joblib
import argparse
import json
import sys
import os
sys.path.append(os.getcwd())
from function import data, model_train, submit, selection

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

#データ取得
#train = feather.read_dataframe('./data/input/target_encoding_train.feather')
#X_test = feather.read_dataframe('./data/input/target_encoding_test.feather')
train, X_test = data.get_base_line_dataframe()

y_train = train[config["target_name"]]
X_train = train.drop([config["target_name"]],axis=1)

#外れ値を除去
X_train, y_train = data.outliers_delete(X_train, y_train, percent=0.01)

#インポータンス0のカラムを削除
X_train, X_test = selection.delete_importance_0(X_train, X_test)

#目的変数を対数化
comment="base_line+log+delete_importance_0"
y_train_pred, y_test_pred, models, score = model_train.run_lgbm(X_train, np.log(y_train), X_test, comment = comment,save=False)


#予測結果を指数化して元に戻す
submit.submit_csv(np.exp(y_test_pred), score)