
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, model_name, mode):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'result/{model_name}/Confusion_Matrix_{mode}.png')
    # plt.show()


def plot_roc_curve(y_true, y_score, model_name, mode, n_classes):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(f'result/{model_name}/ROC_{mode}.png')
    # plt.show()


def plot_pr_curve(y_true, y_score, model_name, mode, n_classes):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    precision, recall, pr_auc = dict(), dict(), dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AUC = {pr_auc[i]:.3f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(f'result/{model_name}/PR_{mode}.png')
    # plt.show()

def plot_comparison(models, metrics):
    metrics_names = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-score']
    num_metrics = len(metrics_names)
    num_models = len(models)
    x = np.arange(num_metrics)

    plt.figure(figsize=(12, 8))
    width = 0.8 / num_models  # 根据模型数量调整宽度

    for i, (model, values) in enumerate(zip(models, metrics)):
        offset = i * width
        bars = plt.bar(x + offset, values, width, label=model)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(x + width * (num_models - 1) / 2, metrics_names)
    plt.ylabel('Scores')
    plt.title('Model Comparison')
    plt.legend()
    plt.savefig(f'result/comparison_result.png')