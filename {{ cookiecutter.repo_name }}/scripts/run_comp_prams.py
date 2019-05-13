# -- coding: utf-8 --
import datetime
import logging
import argparse
import json
import sys
import os
sys.path.append(os.getcwd())
from models import lgbm
from train_predict import train

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H:%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = config['features']
logging.debug("features : {}".format(feats))

target_name = config['target_name']


def run_comp_prams(df_before, df_after, y):
    before_columns = df_before.columns.values.to_list()
    after_columns = df_after.columns.values.to_list()
    logging.debug("befores features : {}".format(before_columns))
    logging.debug("before train shape : {}".format(df_before.shape))
    logging.debug("after features : {}".format(after_columns))
    logging.debug("after train shape : {}".format(df_after.shape))
    different_columns = list(set(before_columns+after_columns))
    logging.debug("different features : {}".format(different_columns))


    model_name="lgbm"
    logging.debug("model {}".format(model_name))
    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 2,
        'learning_rate': 0.5,
        'num_iteration': 100,
        "max_dapth" : 3,
        "randam_state" : 0
    }
    logging.debug("model params : {}".format(lgbm_params))

    #予測モデルオブジェクト
    lgbm_model_object = lgbm.Lgbm()

    #訓練
    logging.debug("=============================== before model ===============================")
    y_train_preds, y_valid_preds, lgbm_model, score = train.train_foldout(df_before, y, lgbm_model_object, lgbm_params, now)
    logging.debug("=============================== after model ===============================")
    y_train_preds, y_valid_preds, lgbm_model, score = train.train_foldout(df_after, y, lgbm_model_object,
                                                                          lgbm_params, now)
