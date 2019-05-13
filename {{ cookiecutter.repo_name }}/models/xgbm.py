import lightgbm as lgb
import logging
import sys
import os
import pandas as pd
from sklearn.externals import joblib
import xgboost as xgb
sys.path.append(os.getcwd())
from logs.logger import log_evaluation_xgbm
from utils.__init__ import timer
import warnings
warnings.filterwarnings('ignore')
class Xgbm:

    def __init__(self):
        self.name="XGBoost"

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, xgbm_params):
        # データセットを生成する
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_valid, label=y_valid)
        evals = [(dtrain, 'train'), (dtest, 'eval')]
        evals_result = {}

        # ロガーの作成
        logger = logging.getLogger('main')
        callbacks = [log_evaluation_xgbm(logger, period=50)]
        with timer("Training"):
            # 上記のパラメータでモデルを学習する
            model = xgb.train(
                xgbm_params,
                dtrain,
                evals=evals,
                evals_result=evals_result,
                num_boost_round=1000,
                # 30 ラウンド経過しても性能が向上しないときは学習を打ち切る
                early_stopping_rounds=50,
                # ログ
                callbacks=callbacks
            )

        y_train_pred = model.predict(xgb.DMatrix(X_train)).reshape(-1,1)
        #検証データの予測
        y_valid_pred = model.predict(xgb.DMatrix(X_valid)).reshape(-1,1)
        # テストデータの予測
        y_pred = model.predict(xgb.DMatrix(X_test)).reshape(-1,1)

        return y_train_pred, y_valid_pred, y_pred, model

    def save_importance(self,xgbm_models, columns, comment=""):
        df_new = pd.DataFrame(index=[], columns=[])
        for i in range(len(xgbm_models)):
            tmp = pd.DataFrame(list(xgbm_models[i].get_fscore().items()),
                               columns=['feature', 'importance'])
            imp = pd.DataFrame(tmp["importance"].values, index=tmp["feature"].values, columns=["importance"])
            df_new = pd.concat([df_new, imp], axis=1)
        df_new = pd.DataFrame(df_new.mean(axis=1), columns=["importance"])
        joblib.dump(df_new, "./features/importances/{}".format(self.name) + "_{}".format(comment), compress=True)