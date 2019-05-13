import json
import argparse
import sys
import os
from collections import OrderedDict
sys.path.append(os.getcwd())
os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config), object_pairs_hook=OrderedDict)
columns_origin=config["features"]

def read_json():
    # インデントありで表示
    print("{}".format(json.dumps(config,indent=4)))

def write_json(overwrite):
    config_orign = open('./configs/default.json', "w")  # 書き込むファイルを開く
    json.dump(overwrite, config_orign, indent=4)

#DataFrameにある特徴量をconfigに追加
def columns_to_json(df, overwrite=False):
    columns = df.columns.values.tolist()
    if  overwrite:
        config["features"] = list(dict.fromkeys(columns))
    else:
        config["features"] = list(dict.fromkeys(columns_origin+columns))
    write_json(config)

#特徴量名(featherファイルの名称)をconfigに追加
def name_to_json(name, overwrite=False):
    if  overwrite:
        config["features"] = list(dict.fromkeys(name))
    else:
        config["features"] = list(dict.fromkeys(columns_origin+name))
    write_json(config)

