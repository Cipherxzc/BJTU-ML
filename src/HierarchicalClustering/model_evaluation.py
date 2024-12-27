import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from ..Dataset.dataset import FashionMNISTDataset
from hierarchical_classifier import HierarchicalClassifier
from hierarchical_clustering import HierarchicalClustering
import pandas as pd
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluation_fn(y, labels):
    return ((1 - adjusted_rand_score(y, labels)) / 2) ** 2 + (1 - normalized_mutual_info_score(y, labels)) ** 2

def save_evaluation_metrics(accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted, filename):
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


def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_prob, class_names, filename):
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_names[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_pr_curve(y_true, y_prob, class_names, filename):
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        plt.plot(recall, precision, label=f"Class {class_names[i]}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.show()
    plt.close()


def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

    # Weighted metrics
    precision_weighted = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted, all_labels, all_predictions, all_probs


def main():
    os.makedirs("../evaluation_results", exist_ok=True)

    # Load test data
    test_data = pd.read_csv("../data/fashion-mnist_test.csv")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_dataset = FashionMNISTDataset(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Load the hierarchical clustering model
    clustering_model = HierarchicalClustering.load_model("../models/hierarchical_clustering_model.pkl")

    # Load the hierarchical classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = HierarchicalClassifier(clustering_model, num_classes=10).to(device)
    classifier.load_model("../models/hierarchical_classifier_model.safetensors", device=device)

    # Evaluate the model
    accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted, all_labels, all_predictions, all_probs = evaluate_model(classifier, test_loader, device)

    # Print and save evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print("Precision by class:", precision)
    print("Recall by class:", recall)
    print("F1-score by class:", f1)
    print(f"Weighted Precision: {precision_weighted:.2f}")
    print(f"Weighted Recall: {recall_weighted:.2f}")
    print(f"Weighted F1-score: {f1_weighted:.2f}")
    save_evaluation_metrics(accuracy, precision, recall, f1, precision_weighted, recall_weighted, f1_weighted,
                            "../evaluation_results/HierarchicalClustering_evaluation_metrics.txt")

    class_names = [f"Class {i}" for i in range(10)]

    plot_confusion_matrix(all_labels, all_predictions, class_names, "../evaluation_results/HierarchicalClustering_confusion_matrix.png")
    plot_roc_curve(all_labels, all_probs, class_names, "../evaluation_results/HierarchicalClustering_ROC.png")
    plot_pr_curve(all_labels, all_probs, class_names, "../evaluation_results/HierarchicalClustering_PR.png")

    # Print metrics for each class
    num_classes = len(precision)
    for i in range(num_classes):
        print(f"Class {i}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1-Score={f1[i]:.2f}")


if __name__ == "__main__":
    main()
