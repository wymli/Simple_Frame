from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error


def get_metric(y_labels, y_preds, metric_type: str):
    if metric_type == "auc":
        auc = roc_auc_score(y_labels, y_preds)
        return auc
    elif metric_type == "acc":
        acc = accuracy_score(y_labels, y_preds)
        return acc
    elif metric_type == "mse":
        mse = mean_squared_error(y_labels , y_preds)
        return mse
    else:
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_labels, y_preds, average="micro")
        if metric_type == "all":
            acc = accuracy_score(y_labels, y_preds)
            auc = roc_auc_score(y_labels, y_preds)
            return precision, recall, fscore, acc, auc
        elif metric_type == "fscore":
            return fscore
        elif metric_type == "precision":
            return precision
        elif metric_type == "recall":
            return recall
    raise Exception("There is no such metric registered")
