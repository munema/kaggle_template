import time
import pandas as pd
from contextlib import contextmanager
import feather
import numpy as np
import requests
import json
import logging


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    logging.debug("Training time : {} s".format(time.time() - t0))

#指定された特徴量の訓練データとテストデータをロード
def load_datasets(feats):
    dfs = [feather.read_dataframe(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [feather.read_dataframe(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test

#訓練データの正解データをロード
def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train

#DataFrameのメモリを節約
def reduce_mem_usage(df, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object' and col_type != 'datetime64[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print_('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

#LINEに通知を送る
def send_line_notification(message):
    line_token = "9IiQzFYm1yh4erTzQWtrK5rjwo6uSKsWGLDea6rGfVE"  # トークン
    endpoint = "https://notify-api.line.me/api/notify"
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)

#スプレッドシートに書き込む
def write_spreadsheet(*args):
    endpoint = "https://script.google.com/macros/s/AKfycbw3TvzVeGqvoyz3x6uozQISqp4qTbE_-YlzKZ1bopdpiZdA5_M/exec" #スプレットシート名
    requests.post(endpoint, json.dumps(args))