import lightgbm as lgb
import logging
import sys
import os
from sklearn import svm
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
sys.path.append(os.getcwd())
from utils.__init__ import timer
import warnings
warnings.filterwarnings('ignore')

class Svr:
    def __init__(self):
        self.name="SVR"

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, svr_params):

        # ロガーの作成
        logger = logging.getLogger('main')
        #標準化
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_valid_std = sc.fit_transform(X_valid)
        X_test_std = sc.fit_transform(X_test)

        with timer("Training"):
            model = svm.SVR(**svr_params)
            model.fit(X_train_std, y_train)

        y_train_pred = model.predict(X_train_std).reshape(-1,1)
        #検証データの予測
        y_valid_pred = model.predict(X_valid_std).reshape(-1,1)
        # テストデータの予測
        y_pred = model.predict(X_test_std).reshape(-1,1)

        return y_train_pred, y_valid_pred, y_pred, model
