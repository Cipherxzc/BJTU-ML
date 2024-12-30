import pandas as pd
from .adaboost_classifier import ShallowDecisionTree, AdaBoost
from ..Dataset.quantized import equal_frequency_binning
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import argparse

def train_adaboost(train_data_path, test_data_path, model_save_path):
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    X_train = train_data.drop("label", axis=1).values
    y_train = train_data["label"].values

    X_test = test_data.drop("label", axis=1).values
    y_test = test_data["label"].values

    # 数据预处理：标准化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # 量化处理
    X_train_quantized = equal_frequency_binning(X_train, n_bins=10)
    X_test_quantized = equal_frequency_binning(X_test, n_bins=10)

    model = AdaBoost(n_estimators=50, max_depth=4)
    model.fit(X_train_quantized, y_train)

    model.save_model(model_save_path)

    y_pred = model.predict(X_test_quantized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Adaboost model.")
    parser.add_argument("--train-data", required=True, help="Path to the training data CSV file.")
    parser.add_argument("--test-data", required=True, help="Path to the testing data CSV file.")
    parser.add_argument("--model-save-path", required=True, help="Path to save the trained model.")
    args = parser.parse_args()

    train_adaboost(args.train_data, args.test_data, args.model_save_path)