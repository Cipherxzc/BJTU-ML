import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering


def load_data(test_data_path):
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values  # Assuming true labels are available for evaluation
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    return X_test, y_test


def evaluate_clustering_model(model, X_test, y_test):
    y_pred = model.fit_predict(X_test)

    silhouette = silhouette_score(X_test, y_pred)
    davies_bouldin = davies_bouldin_score(X_test, y_pred)
    rand_index = adjusted_rand_score(y_test, y_pred)  # Only if true labels are available

    return silhouette, davies_bouldin, rand_index


def print_evaluation_metrics(silhouette, davies_bouldin, rand_index,
                             filename="HierarchicalClustering_evaluation_metrics.txt"):
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
    print(f"Adjusted Rand Index: {rand_index:.2f}")

    with open(filename, 'w') as f:
        f.write(f"Silhouette Score: {silhouette:.2f}\n")
        f.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}\n")
        f.write(f"Adjusted Rand Index: {rand_index:.2f}\n")


def main():
    X_test, y_test = load_data("../../data/features1_test.csv")

    model = AgglomerativeClustering(n_clusters=10, linkage='ward')

    silhouette, davies_bouldin, rand_index = evaluate_clustering_model(model, X_test, y_test)
    print_evaluation_metrics(silhouette, davies_bouldin, rand_index)


if __name__ == "__main__":
    main()
