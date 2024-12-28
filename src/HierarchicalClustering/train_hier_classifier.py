import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from .hierarchical_classifier import HierarchicalClassifier
from ..Dataset.dataset import FeaturesDataset


def train_hier_classifier(train_data_path, test_data_path, model_save_path):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")


    train_dataset = FeaturesDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalClassifier(num_classes=10, feature_dim=train_dataset.feature_dim).to(device)
    model.fit(train_loader, epochs=10, device=device)
    

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    y_proba = model.predict_proba(X_test)
    print(f'Predicted probabilities for the first test sample: {y_proba[0]}')

    model.save_model(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NaiveBayes model.")
    parser.add_argument("--train-data", required=True, help="Path to the training data CSV file.")
    parser.add_argument("--test-data", required=True, help="Path to the testing data CSV file.")
    parser.add_argument("--model-save-path", required=True, help="Path to save the trained model.")
    args = parser.parse_args()

    train_hier_classifier(args.train_data, args.test_data, args.model_save_path)