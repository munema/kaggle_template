from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,log_loss,mean_squared_error,mean_squared_log_error
import numpy as np

def loss(y_true, y_pred, metrics):
    metrics=metrics.lower()
    if metrics=="accuracy":
        if len(y_pred.shape[1]>1):y_pred=np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)
    elif metrics=="f1":
        return f1_score(y_true, y_pred)
    elif metrics=="auc":
        return roc_auc_score(y_true, y_pred)
    elif metrics=="l2":
        return mean_squared_error(y_true, y_pred)
    elif metrics=="l2_root":
        return mean_squared_log_error(y_true, y_pred)
    else: return log_loss(y_true, y_pred)