import os
import json
import numpy as np
import argparse
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.Adaboost.adaboost_classifier import AdaBoost
from src.HierarchicalClustering.hierarchical_classifier import HierarchicalClassifier
from src.Linear.linear_pca_classifier import SoftmaxRegression
from src.NaiveBayes.naive_bayes import NaiveBayes
from src.Dataset.quantized import equal_frequency_binning
from utils import plot_confusion_matrix, plot_roc_curve, plot_pr_curve, plot_comparison


def no_transform(X):
    return X


model_info = {
    "Adaboost": {
        "save_path": ["models/adaboost_model_1.pkl", "models/adaboost_model_2.pkl"],
        "model": AdaBoost,
        "transform": equal_frequency_binning
    },
    "HierarchicalClustering": {
        "save_path": ["models/hierarchical_clustering_model_1.safetensors", "models/hierarchical_clustering_model_2.safetensors"],
        "model": HierarchicalClassifier,
        "transform": no_transform
    },
    "Linear": {
        "save_path": ["models/linear_1", "models/linear_2"],
        "model": SoftmaxRegression,
        "transform": no_transform
    },
    "NaiveBayes": {
        "save_path": ["models/naive_bayes_model_1.safetensors", "models/naive_bayes_model_2.safetensors"],
        "model": NaiveBayes,
        "transform": no_transform
    },
}


def evaluate_model(model_name, mode):
    test_data = f"data/features{mode}_test.csv"
    save_path = model_info[model_name]["save_path"][int(mode)-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_info[model_name]["model"]()
    if hasattr(model, 'to'):
        model = model.to(device)

    model.load_model(save_path)

    test_data = pd.read_csv(test_data)
    print(f"Test data shape: {test_data.shape}")

    X = test_data.iloc[:, :-1].values
    y = test_data.iloc[:, -1].values

    X = model_info[model_name]["transform"](X)

    
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)


    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average=None)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')

    results = {
        'Accuracy': round(accuracy, 3),
        'Weighted Precision': round(weighted_precision, 3),
        'Weighted Recall': round(weighted_recall, 3),
        'Weighted F1-score': round(weighted_f1, 3),
        'Precision by class': {str(i): round(p, 3) for i, p in enumerate(precision)},
        'Recall by class': {str(i): round(r, 3) for i, r in enumerate(recall)},
        'F1-score by class': {str(i): round(f, 3) for i, f in enumerate(f1)}
    }

    dir_path = os.path.join('result', model_name)
    os.makedirs(dir_path, exist_ok=True)
    with open(f'result/{model_name}/evaluation_metrics_{mode}.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Print results to console
    print(json.dumps(results, indent=4))

    # Visualize and save plots
    plot_confusion_matrix(y, y_pred, model_name, mode)
    plot_roc_curve(y, y_score, model_name, mode, n_classes=len(np.unique(y)))
    plot_pr_curve(y, y_score, model_name, mode, n_classes=len(np.unique(y)))

    return accuracy, weighted_precision, weighted_recall, weighted_f1
    

def main():
    available_models = ["Adaboost", "HierarchicalClustering", "Linear", "NaiveBayes"]

    parser = argparse.ArgumentParser(description="Select models for training.")
    parser.add_argument(
        "-m", "--models",
        nargs='+',
        choices=[f"{model}1" for model in available_models] + [f"{model}2" for model in available_models],
        required=True,
        help="Select one or more models and modes to evaluate. For example, Adaboost1 or Adaboost2."
    )
    args = parser.parse_args()

    metrics = []
    for model_mode in args.models:
        selected_model = model_mode[:-1]
        mode = model_mode[-1]
        metrics.append(evaluate_model(selected_model, mode))

    plot_comparison(args.models, metrics)


if __name__ == "__main__":
    main()