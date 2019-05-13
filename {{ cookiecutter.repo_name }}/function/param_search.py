from sklearn.model_selection import GridSearchCV
import json
import argparse
import xgboost as xgb

def gridserch_cv(X_train, y_train, tuned_params ,model):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/default.json')
    options = parser.parse_args()
    config = json.load(open(options.config))

    clf = GridSearchCV(
        estimator=model, # 識別器
        param_grid=tuned_params, # 最適化したいパラメータセット
        cv=config["K-Fold"]["n_splits"],
        scoring = config["loss"]
    )

    clf.fit(X_train, y_train, random_state=config["K-Fold"]["random_state"])

    return clf