import numpy as np
import pandas as pd
from .naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
import argparse

def train_naive_bayes(train_data_path, test_data_path, model_save_path):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # 归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = NaiveBayes()
    model.fit(X_train, y_train)

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

    train_naive_bayes(args.train_data, args.test_data, args.model_save_path)