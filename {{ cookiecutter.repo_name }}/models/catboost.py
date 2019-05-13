import lightgbm as lgb
import logging
import sys
import os
from catboost import CatBoost
from catboost import Pool
from function import data
sys.path.append(os.getcwd())
from utils.__init__ import timer
import warnings
warnings.filterwarnings('ignore')

class Catboost:

    def __init__(self):
        self.name="CatBoost"

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, catboost_params):
        # データセットを生成する
        train_pool = Pool(X_train, label=y_train)
        valid_pool = Pool(X_valid, label=y_valid)
        evals = [valid_pool]
        model = CatBoost(catboost_params)

        # ロガーの作成
        with timer("Training"):
            # 上記のパラメータでモデルを学習する
            model.fit(
                train_pool,
                cat_features= data.get_category_columns(X_train),
                eval_set=evals,
                # 50 ラウンド経過しても性能が向上しないときは学習を打ち切る
                early_stopping_rounds=50,
            )

        y_train_pred = model.predict(train_pool).reshape(-1,1)
        #検証データの予測
        y_valid_pred = model.predict(valid_pool).reshape(-1,1)

        # テストデータの予測
        y_pred = model.predict(X_test).reshape(-1,1)

        return y_train_pred, y_valid_pred, y_pred, model