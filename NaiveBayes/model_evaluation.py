import os
import numpy as np
import pandas as pd
from naive_bayes import NaiveBayes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def load_data(test_data_path):
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    X_test = X_test / 255.0  # Normalize
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted

def print_evaluation_metrics(accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted,
                             filename="../evaluation_results/NaiveBayes_evaluation_metrics.txt"):
    print(f"Accuracy: {accuracy:.2f}")
    print("Precision by class:", precision)
    print("Recall by class:", recall)
    print("F1-score by class:", f1)

    print(f"Weighted Precision: {precision_weighted:.2f}")
    print(f"Weighted Recall: {recall_weighted:.2f}")
    print(f"Weighted F1-score: {f1_weighted:.2f}")

    with open(filename, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Weighted Precision: {precision_weighted:.2f}\n")
        f.write(f"Weighted Recall: {recall_weighted:.2f}\n")
        f.write(f"Weighted F1-score: {f1_weighted:.2f}\n")
        f.write("Precision by class:\n")
        f.write(", ".join([f"{i}: {p:.2f}" for i, p in enumerate(precision)]) + "\n")
        f.write("Recall by class:\n")
        f.write(", ".join([f"{i}: {r:.2f}" for i, r in enumerate(recall)]) + "\n")
        f.write("F1-score by class:\n")
        f.write(", ".join([f"{i}: {f1_score:.2f}" for i, f1_score in enumerate(f1)]) + "\n")

def plot_confusion_matrix(y_test, y_pred, class_names, filename="../evaluation_results/NaiveBayes_confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_roc_curve(y_test, y_score, class_names, filename="../evaluation_results/NaiveBayes_ROC.png"):
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_pr_curve(y_test, y_score, class_names, filename="../evaluation_results/NaiveBayes_PR.png"):
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.show()
    plt.close()

def main():
    os.makedirs("../evaluation_results", exist_ok=True)

    X_test, y_test = load_data("../data/fashion-mnist_test.csv")
    model_NaiveBayes = NaiveBayes.load_model("../models/naive_bayes_model.pkl")

    accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted = evaluate_model(model_NaiveBayes, X_test, y_test)
    print_evaluation_metrics(accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted)

    num_classes = len(np.unique(y_test))
    class_names = [f'Class {i}' for i in range(num_classes)]
    y_score = model_NaiveBayes.predict_proba(X_test)

    plot_confusion_matrix(y_test, model_NaiveBayes.predict(X_test), class_names)
    plot_roc_curve(y_test, y_score, class_names)
    plot_pr_curve(y_test, y_score, class_names)

    for i in range(num_classes):
        print(f"Class {i}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-Score={f1[i]:.2f}")

if __name__ == "__main__":
    main()
