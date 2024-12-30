import argparse
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from src.Adaboost.adaboost_classifier import AdaBoost
from src.HierarchicalClustering.hierarchical_classifier import HierarchicalClassifier
from src.Linear.linear_pca_classifier import SoftmaxRegression
from src.NaiveBayes.naive_bayes import NaiveBayes
from src.Dataset.quantized import equal_frequency_binning

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
    y_proba = model.predict_proba(X)

    print(y)
    print(y_pred)
    # print(y_proba)


    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    

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

    for model_mode in args.models:
        selected_model = model_mode[:-1]
        mode = model_mode[-1]
        evaluate_model(selected_model, mode)


if __name__ == "__main__":
    main()