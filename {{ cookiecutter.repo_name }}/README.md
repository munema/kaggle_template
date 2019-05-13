# 使用手順
1. input.csv -> input.feather
```
python scripts/convert_to_feather.py
```
2. 前処理
```
python features/preprocessing.py
```
3. 特徴量作成
```
python features/create.py
```
4. 生成した特徴量を組み合わせたDataframe作成
```
python features/training_data.py
```
5. 学習+予測
```
python run/run_base_line.py
```

## ファイル説明
### Config
```
configs/default.json
```
### 特徴量生成
```
features/create.py
```
### 前処理
```
features/preprocessing.py
```
### 生成した特徴量を組み合わせる
```
features/training_data.py
```
### 様々な関数定義
```
function/data.py
```
### 特徴量選択についての関数定義
```
function/submit.py
```
### submitファイル作成
```
function/selection.py
```
### 特徴量選択についての関数定義
```
function/selection.py
```
### モデルの定義
```
models/
```
### 学習
```
run/run_base_line.py
```
### input.csv -> input.feather
```
scripts/convert_to_feather.py
```
### lossの定義
```
train_predict/loss.py
```

## 学習の流れ
```
run/run_base_line.py -> function/model_train.py -> train_predict/train.py
```
