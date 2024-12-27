import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data_path = "../data/fashion-mnist_train.csv"
test_data_path = "../data/fashion-mnist_test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values

# 数据预处理：标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

from adaboost_classifier import ShallowDecisionTree, AdaBoost

base_model = ShallowDecisionTree
model = AdaBoost(base_model, n_estimators=50, max_depth=4)
model.fit(X_train, y_train)

model.save_model("../models/adaboost_model.pkl")

from adaboost_classifier import ShallowDecisionTree, AdaBoost
from sklearn.metrics import accuracy_score

model = AdaBoost.load_model("../models/adaboost_model.pkl")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')