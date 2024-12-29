import pandas as pd
from .linear_pca_classifier import SoftmaxRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse

def train_linear(train_data_path, test_data_path, model_save_path):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    X_train = train_data.drop("label", axis=1).values
    y_train = train_data["label"].values

    X_test = test_data.drop("label", axis=1).values
    y_test = test_data["label"].values

    # 数据标准化：将数据标准化到均值为0，方差为1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用 PCA 进行降维
    pca = PCA(n_components=50)  # 保留50个主成分
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = SoftmaxRegression(learning_rate=0.001, iterations=1000)
    model.fit(X_train_pca, y_train, num_classes=10)

    model.save_model(model_save_path)

    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Liner model.")
    parser.add_argument("--train-data", required=True, help="Path to the training data CSV file.")
    parser.add_argument("--test-data", required=True, help="Path to the testing data CSV file.")
    parser.add_argument("--model-save-path", required=True, help="Path to save the trained model.")
    args = parser.parse_args()

    train_linear(args.train_data, args.test_data, args.model_save_path)
