import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import FashionMNISTDataset
from simclr import SimCLR
from simclr_classifier import SimCLRClassifier
from torchvision import transforms


def evaluate_model(model, X_test, y_test, batch_size=16):
    model.eval()
    num_samples = len(X_test)
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            X_batch = torch.tensor(X_test[i:i + batch_size]).float().to(next(model.parameters()).device)
            y_batch = torch.tensor(y_test[i:i + batch_size]).to(next(model.parameters()).device)

            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Weighted metrics
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted, y_true, y_pred, y_probs


def print_evaluation_metrics(accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted,
                             filename="../evaluation_results/SimCLR_evaluation_metrics.txt"):
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


def plot_confusion_matrix(y_true, y_pred, class_names, filename="../evaluation_results/SimCLR_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_probs, class_names, filename="../evaluation_results/SimCLR_ROC.png"):
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_pr_curve(y_true, y_probs, class_names, filename="../evaluation_results/SimCLR_PR.png"):
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        average_precision = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP = {average_precision:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.show()
    plt.close()


def main():
    os.makedirs("../evaluation_results", exist_ok=True)

    # Load test data
    test_data_path = "../data/fashion-mnist_test.csv"
    simclr_model_path = "../models/simclr_model.safetensors"
    classifier_model_path = "../models/simclr_classifier_model.safetensors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_data = pd.read_csv(test_data_path)
    test_dataset = FashionMNISTDataset(test_data, transform=transform)
    X_test = np.stack([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

    # Load models
    base_model = SimCLR(out_dim=128)
    base_model.load_model(simclr_model_path, device=device)

    classifier = SimCLRClassifier(base_model=base_model, num_classes=10).to(device)
    classifier.load_model(classifier_model_path, device=device)

    accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted, y_true, y_pred, y_probs = evaluate_model(classifier, X_test, y_test)

    print_evaluation_metrics(accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted,
                             filename="../evaluation_results/SimCLR_evaluation_metrics.txt")

    class_names = [f"Class {i}" for i in range(10)]

    plot_confusion_matrix(y_true, y_pred, class_names, "../evaluation_results/SimCLR_confusion_matrix.png")
    plot_roc_curve(y_true, y_probs, class_names, "../evaluation_results/SimCLR_ROC.png")
    plot_pr_curve(y_true, y_probs, class_names, "../evaluation_results/SimCLR_PR.png")

    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-Score={f1[i]:.2f}")


if __name__ == "__main__":
    main()
