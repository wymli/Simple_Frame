from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error


def get_metric(y_labels, y_preds, metric_mode: str):
    if metric_mode == "auc":
        auc = roc_auc_score(y_labels, y_preds)
        return auc
    elif metric_mode == "acc":
        acc = accuracy_score(y_labels, y_preds)
        return acc
    elif metric_mode == "mse":
        mse = mean_squared_error(y_labels , y_preds)
        return mse
    else:
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_labels, y_preds, average="micro")
        if metric_mode == "all":
            acc = accuracy_score(y_labels, y_preds)
            auc = roc_auc_score(y_labels, y_preds)
            return precision, recall, fscore, acc, auc
        elif metric_mode == "fscore":
            return fscore
        elif metric_mode == "precision":
            return precision
        elif metric_mode == "recall":
            return recall
    raise Exception("There is no such metric registered")
