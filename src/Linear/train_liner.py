import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_data_path = "../data/fashion-mnist_train.csv"
test_data_path = "../data/fashion-mnist_test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

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

print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

from linear_pca_classifier import SoftmaxRegression

model = SoftmaxRegression(learning_rate=0.001, iterations=1000)
model.fit(X_train, y_train, num_classes=10)

model.save_model("../models/linear_pca_model.pkl")

from sklearn.metrics import accuracy_score

model = SoftmaxRegression.load_model("../models/linear_pca_model.pkl")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')