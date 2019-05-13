# -- coding: utf-8 --
import datetime
import json
import sys
import os
sys.path.append(os.getcwd())
from models import lgbm, xgbm, lasso, svm, catboost
from train_predict import train
import importlib


def run_lgbm(X_train, y_train, X_test, comment = "", save=False):
    now = datetime.datetime.now()
    import logging
    importlib.reload(logging)
    logging.basicConfig(
        filename='./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now), level=logging.DEBUG
    )

    config = json.load(open("configs/default.json"))

    logging.debug('./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now))
    logging.debug('--------------------------------------------------------------------')
    logging.debug("comment : {}".format(comment))
    logging.debug('--------------------------------------------------------------------')
    feats = X_train.columns.values
    logging.debug("features : {}".format(feats))

    logging.debug("train shape : {}".format(X_train.shape))

    logging.debug('============================================================================================')

    # 予測モデルオブジェクト
    model_object = lgbm.Lgbm()

    logging.debug("model {}".format(model_object.name))
    params = config['lgbm_params']

    # random_stateの設定
    seed = 0
    params["random_state"] = seed
    logging.debug("params : {}".format(params))

    #訓練
    y_train_preds, y_test_preds, models, score = train.train_cv(X_train, y_train, X_test, model_object, params, now, comment, save=save)

    # インポータンスを保存
    model_object.save_importance(models, X_train.columns, comment)

    return y_train_preds, y_test_preds, models, score



def run_xgbm(X_train, y_train, X_test, comment = "", save=False):
    now = datetime.datetime.now()
    import logging
    importlib.reload(logging)
    logging.basicConfig(
        filename='./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now), level=logging.DEBUG
    )

    config = json.load(open("configs/default.json"))

    logging.debug('./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now))
    logging.debug('--------------------------------------------------------------------')
    logging.debug("comment : {}".format(comment))
    logging.debug('--------------------------------------------------------------------')
    feats = X_train.columns.values
    logging.debug("features : {}".format(feats))

    logging.debug("train shape : {}".format(X_train.shape))

    logging.debug('============================================================================================')

    # 予測モデルオブジェクト
    model_object = xgbm.Xgbm()

    logging.debug("model {}".format(model_object.name))
    params = config['xgbm_params']

    # random_stateの設定
    seed = 0
    params["random_state"] = seed
    logging.debug("params : {}".format(params))

    #訓練
    y_train_preds, y_test_preds, models, score = train.train_cv(X_train, y_train, X_test, model_object, params, now, comment, save=save)

    # インポータンス保存
    model_object.save_importance(models, X_train.columns, comment)

    return y_train_preds, y_test_preds, models, score



def run_catboost(X_train, y_train, X_test, comment = "", save=False):
    now = datetime.datetime.now()
    import logging
    importlib.reload(logging)
    logging.basicConfig(
        filename='./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now), level=logging.DEBUG
    )

    config = json.load(open("configs/default.json"))

    logging.debug('./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now))
    logging.debug('--------------------------------------------------------------------')
    logging.debug("comment : {}".format(comment))
    logging.debug('--------------------------------------------------------------------')
    feats = X_train.columns.values
    logging.debug("features : {}".format(feats))

    logging.debug("train shape : {}".format(X_train.shape))

    logging.debug('============================================================================================')

    # 予測モデルオブジェクト
    model_object = catboost.Catboost()

    logging.debug("model {}".format(model_object.name))
    #params = config['_params']
    params=[]

    # random_stateの設定
    #seed = 0
    #params["random_state"] = seed
    #logging.debug("params : {}".format(params))

    #訓練
    y_train_preds, y_test_preds, models, score = train.train_cv(X_train, y_train, X_test, model_object, params, now, comment, save=save)

    return y_train_preds, y_test_preds, models, score



def run_lasso(X_train, y_train, X_test, comment = "", save=False):
    now = datetime.datetime.now()
    import logging
    importlib.reload(logging)
    logging.basicConfig(
        filename='./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now), level=logging.DEBUG
    )

    config = json.load(open("configs/default.json"))

    logging.debug('./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now))
    logging.debug('--------------------------------------------------------------------')
    logging.debug("comment : {}".format(comment))
    logging.debug('--------------------------------------------------------------------')
    feats = X_train.columns.values
    logging.debug("features : {}".format(feats))

    logging.debug("train shape : {}".format(X_train.shape))

    logging.debug('============================================================================================')

    # 予測モデルオブジェクト
    model_object = lasso.Lasso_regesstion()

    logging.debug("model {}".format(model_object.name))
    params = config['lasso_params']

    # random_stateの設定
    seed = 0
    params["random_state"] = seed
    logging.debug("params : {}".format(params))

    #訓練
    y_train_preds, y_test_preds, models, score = train.train_cv(X_train, y_train, X_test, model_object, params, now, comment, save=save)

    return y_train_preds, y_test_preds, models, score


def run_svr(X_train, y_train, X_test, comment = "", save=False):
    now = datetime.datetime.now()
    import logging
    importlib.reload(logging)
    logging.basicConfig(
        filename='./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now), level=logging.DEBUG
    )

    config = json.load(open("configs/default.json"))

    logging.debug('./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now))
    logging.debug('--------------------------------------------------------------------')
    logging.debug("comment : {}".format(comment))
    logging.debug('--------------------------------------------------------------------')
    feats = X_train.columns.values
    logging.debug("features : {}".format(feats))

    logging.debug("train shape : {}".format(X_train.shape))

    logging.debug('============================================================================================')

    # 予測モデルオブジェクト
    model_object = svm.Svr()

    logging.debug("model {}".format(model_object.name))
    params = config['svr_params']

    #訓練
    y_train_preds, y_test_preds, models, score = train.train_cv(X_train, y_train, X_test, model_object, params, now, comment, save=save)

    return y_train_preds, y_test_preds, models, score