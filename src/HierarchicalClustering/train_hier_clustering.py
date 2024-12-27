import numpy as np
import pandas as pd
from hierarchical_clustering import HierarchicalClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

train_data = pd.read_csv("../data/fashion-mnist_train.csv")
test_data = pd.read_csv("../data/fashion-mnist_test.csv")

X_train = train_data.iloc[:1000, 1:].values
y_train = train_data.iloc[:1000, 0].values
X_test = test_data.iloc[:500, 1:].values
y_test = test_data.iloc[:500, 0].values

# 归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


def evaluation_fn(y, labels):
    return ((1 - adjusted_rand_score(y, labels)) / 2)**2 + (1 - normalized_mutual_info_score(y, labels))**2

model = HierarchicalClustering(evaluation_fn)
model.fit(X_train, y_train)

model.save_model("../models/hierarchical_clustering_model.pkl")


model = HierarchicalClustering.load_model("../models/hierarchical_clustering_model.pkl")

labels = model.predict(X_test)

ari = adjusted_rand_score(y_test, labels)
nmi = normalized_mutual_info_score(y_test, labels)

print(f'Adjusted Rand Index: {ari:.2f}')
print(f'Normalized Mutual Information: {nmi:.2f}')