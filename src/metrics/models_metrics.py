import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc,
                             confusion_matrix, cohen_kappa_score)


def print_sklearn_model_metrics(model, x_test: np.ndarray, y_test: np.ndarray, multiclass: bool = False):
    y_pred = model.predict(x_test)

    _get_accuracy(y_test, y_pred)
    print('Cohen-Kappa:', cohen_kappa_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_pred_proba = model.predict_proba(x_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

    print("AUC-ROC:", auc_roc)
    plot_confusion_matrix(y_test, y_pred)


def _get_accuracy(y_test: np.ndarray, y_pred: np.ndarray):
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)



def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_pr_curve(y_test: np.ndarray, y_pred: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_test, y_pred[:, 0])
    auc_pr = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % auc_pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_pr_curve_sklearn(y_test: np.ndarray, y_pred: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auc_pr = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % auc_pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_roc_curve(y_test: np.ndarray, y_pred: np.ndarray):
    fpr, tpr, _ = roc_curve(y_test, y_pred[:, 0])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curve_sklearn(y_test: np.ndarray, y_pred: np.ndarray):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()