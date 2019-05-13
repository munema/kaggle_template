# -- coding: utf-8 --
import pandas as pd
import numpy as np
import datetime
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

#予測と, 検証スコア
def submit_csv(y_pred, score):
    now = datetime.datetime.now()
    # submitファイルの作成
    ID_name = config['ID_name']
    sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])


    sub[config['target_name']] = y_pred

    sub.to_csv(
        './data/output/sub_{0:%Y%m%d%H:%M%S}_{1}.csv'.format(now, score),
        index=False
    )