# -- coding: utf-8 --
import json
import argparse
import os
import sys
import numpy as np
import logging
from sklearn.externals import joblib
sys.path.append(os.getcwd())
from logs.logger import log_best_xgbm, log_best_lgbm
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from utils.__init__ import reduce_mem_usage,send_line_notification,write_spreadsheet
from train_predict.loss import loss
if sys.argv:
        del sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))
metric = config["loss"]

def train_cv(X_train, y_train, x_test, model_object, params, now, comment, save=False):
    cv = config["cv"]

    if cv=="kfold" or cv=="skfold":
        if  cv=="kfold":
            kf = KFold(**config["kfold"])
        elif cv=="skfold":
            kf = StratifiedKFold(**config["skfold"])

        ntrain = X_train.shape[0]
        ntest = x_test.shape[0]
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((config[cv]["n_splits"], ntest))
        train_loss = []
        valid_loss = []
        models = []
        i=0
        for train_index, valid_index in kf.split(X_train):
            logging.debug('-------------------CV {}-------------------'.format(i+1))
            X_tr = X_train.iloc[train_index, :]
            X_va = X_train.iloc[valid_index, :]
            y_tr = y_train[train_index]
            y_va = y_train[valid_index]

            # モデルの実行
            y_tr_pred, y_va_pred, y_test_pred, model = model_object.train_and_predict(
                X_tr, X_va, y_tr, y_va, x_test, params
            )

            # 結果の保存
            oof_train[valid_index] = y_va_pred.reshape(-1,)
            oof_test_skf[i, :] = y_test_pred.reshape(-1,)

            train_loss.append(loss(y_tr, y_tr_pred, metric))
            valid_loss.append(loss(y_va, y_va_pred, metric))
            models.append(model)

            # ベストスコア
            if  model_object.name == "LightGBM":
                log_best_lgbm(model, metric)
            elif    model_object.name == "XGBoost":
                log_best_xgbm(model, metric)

            i+=1

        oof_test[:] = oof_test_skf.mean(axis=0)

        # CVスコア
        valid_score = sum(valid_loss) / len(valid_loss)
        logging.debug('-------------------CV scores-------------------')
        logging.debug("train loss : {}".format(train_loss))
        logging.debug("valid loss : {}".format(valid_loss))
        logging.debug("average valid loss : {}".format(valid_score))

        # 保存
        if save:
            # スプレッドシートにスコアを書き込み
            write_spreadsheet(now.strftime('%Y%m%d%H:%M%S'), model_object.name,X_train.shape[1],comment,valid_score)

            #Lineに通知
            send_line_notification("model : {},\n\n comment : {},\n\n shape : {},\n\n params : {},\n\n scores{}".format(model_object.name,comment, X_train.shape, params, valid_score))

            # 保存
            dir_path="./logs/log_{}".format(comment)+"_{0:%Y%m%d%H:%M%S}".format(now)
            os.mkdir(dir_path)
            joblib.dump(X_train, '{}/X_train'.format(dir_path), compress=True)
            joblib.dump(models, '{}/{}'.format(dir_path,model_object.name), compress=True)
            joblib.dump(oof_train, '{}/oof_train'.format(dir_path), compress=True)
            joblib.dump(oof_test, '{}/oof_test'.format(dir_path), compress=True)

        return oof_train, oof_test, models, valid_score

    #Hold Out
    else:
        X_tr, y_tr, X_va, y_va = train_test_split(X_train, y_train, **config["holdout"])
        # モデルの実行
        y_tr_pred, y_va_pred, y_test_pred, model = model_object.train_and_predict(
            X_tr, X_va, y_tr, y_va, x_test, params, proba=False
        )
        train_loss = loss(y_tr, y_tr_pred)
        valid_loss = loss(y_va, y_va_pred)

        logging.debug('-------------------Fold Out scores-------------------')
        logging.debug("train loss : {}".format(train_loss))
        logging.debug("valid loss : {}".format(valid_loss))

        # 保存
        if save:
            # スプレッドシートにスコアを書き込み
            write_spreadsheet(now.strftime('%Y%m%d%H:%M%S'), model_object.name, X_train.shape[1], comment, valid_loss)

            # Lineに通知
            send_line_notification(
                "model : {},\n\n comment : {},\n\n shape : {},\n\n params : {},\n\n scores{}".format(model_object.name,
                                                                                                     comment,
                                                                                                     X_train.shape,
                                                                                                     params,
                                                                                                     valid_loss))

            # 保存
            dir_path = "./logs/log_{}".format(comment) + "_{0:%Y%m%d%H:%M%S}".format(now)
            os.mkdir(dir_path)
            joblib.dump(X_train, '{}/X_train'.format(dir_path), compress=True)
            joblib.dump(model, '{}/{}'.format(dir_path, model_object.name), compress=True)
            joblib.dump(y_tr_pred, '{}/y_train_preds'.format(dir_path), compress=True)
            joblib.dump(y_va_pred, '{}/y_valid_preds'.format(dir_path), compress=True)

        return y_tr_pred, y_test_pred, model, valid_loss


