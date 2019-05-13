import lightgbm as lgb
import logging
import sys
import os
import pandas as pd
from sklearn.externals import joblib
sys.path.append(os.getcwd())
from logs.logger import log_evaluation_lgbm
from utils.__init__ import timer
import warnings
warnings.filterwarnings('ignore')

class Lgbm:

    def __init__(self):
        self.name="LightGBM"

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, lgbm_params):
        # データセットを生成する
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        evals_result = {}

        # ロガーの作成
        logger = logging.getLogger('main')
        callbacks = [log_evaluation_lgbm(logger, period=50)]
        with timer("Training"):
            # 上記のパラメータでモデルを学習する
            model = lgb.train(
                lgbm_params, lgb_train,
                # モデルの評価用データを渡す
                valid_sets=lgb_eval,
                evals_result=evals_result,
                # 最大で 1000 ラウンドまで学習する
                num_boost_round=1000,
                # 30 ラウンド経過しても性能が向上しないときは学習を打ち切る
                early_stopping_rounds=50,
                # ログ
                callbacks=callbacks
            )

        y_train_pred = model.predict(X_train, num_iteration=model.best_iteration).reshape(-1,1)
        #検証データの予測
        y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration).reshape(-1,1)
        # テストデータの予測
        y_pred = model.predict(X_test, num_iteration=model.best_iteration).reshape(-1,1)

        return y_train_pred, y_valid_pred, y_pred, model


    def save_importance(self,lgbm_models, columns, comment=""):
        df_new = pd.DataFrame(index=[], columns=[])
        for i in range(len(lgbm_models)):
            imp = pd.DataFrame(lgbm_models[i].feature_importance(), index=columns, columns=['importance'])
            df_new = pd.concat([df_new, imp], axis=1)
        df_new = pd.DataFrame(df_new.mean(axis=1), columns=["importance"])
        joblib.dump(df_new, "./features/importances/{}".format(self.name)+"_{}".format(comment), compress=True)