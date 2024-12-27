import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pickle

class HierarchicalClustering:
    def __init__(self, evaluation_fn=None):
        self.distance_metric = None
        self.clusters = None
        self.evaluation_fn = evaluation_fn

    def fit(self, X, y):
        distance_metrics = ['euclidean', 'cityblock', 'chebyshev']
        best_loss = np.inf
        best_distance_metric = None
        best_clusters = None
        best_labels = None

        for distance_metric in distance_metrics:
            for clusters in range(9, 12):
                self.distance_metric = distance_metric
                self.clusters = clusters

                labels = self.predict(X)
                loss = self.evaluation_fn(y, labels)
                print(f"distance_metrics: {distance_metric}, clusters: {clusters}")
                print(f'Loss: {loss:.4f}')

                if loss < best_loss:
                    best_loss = loss
                    best_distance_metric = distance_metric
                    best_clusters = clusters
                    best_labels = labels

        self.distance_metric = best_distance_metric
        self.clusters = best_clusters
        self.labels_ = best_labels

        print(f"distance_metrics: {best_distance_metric}, clusters: {best_clusters}")
        print(f'Loss: {best_loss:.4f}')

    def predict(self, X):
        distance_matrix = squareform(pdist(X, metric=self.distance_metric))
        clusters = [[i] for i in range(len(X))]
        cluster_sizes = [1] * len(X)

        while len(clusters) > self.clusters:
            cluster1, cluster2 = self._find_closest_clusters(distance_matrix)
            clusters = self._merge_clusters(clusters, cluster1, cluster2)
            distance_matrix = self._update_distance_matrix(distance_matrix, cluster1, cluster2, cluster_sizes)
            cluster_sizes.append(cluster_sizes[cluster1] + cluster_sizes[cluster2])
            cluster_sizes = [cluster_sizes[i] for i in range(len(cluster_sizes)) if i != cluster1 and i != cluster2]
        
        labels = np.zeros(X.shape[0], dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_idx
        return labels

    def _find_closest_clusters(self, distance_matrix):
        min_distance = np.inf
        closest_clusters = None
        for i in range(len(distance_matrix)):
            for j in range(i + 1, len(distance_matrix)):
                if distance_matrix[i, j] < min_distance:
                    min_distance = distance_matrix[i, j]
                    closest_clusters = (i, j)
        return closest_clusters

    def _merge_clusters(self, clusters, cluster1, cluster2):
        new_cluster = clusters[cluster1] + clusters[cluster2]
        new_clusters = [clusters[i] for i in range(len(clusters)) if i != cluster1 and i != cluster2]
        new_clusters.append(new_cluster)
        return new_clusters

    def _update_distance_matrix(self, distance_matrix, cluster1, cluster2, cluster_sizes):
        new_distance_matrix = np.delete(distance_matrix, (cluster1, cluster2), axis=0)
        new_distance_matrix = np.delete(new_distance_matrix, (cluster1, cluster2), axis=1)
        
        new_distances = []
        for i in range(len(distance_matrix)):
            if i == cluster1 or i == cluster2:
                continue
            new_distance = np.sqrt(
                (cluster_sizes[cluster1] + cluster_sizes[i]) * distance_matrix[cluster1, i]**2 +
                (cluster_sizes[cluster2] + cluster_sizes[i]) * distance_matrix[cluster2, i]**2 -
                cluster_sizes[i] * distance_matrix[cluster1, cluster2]**2
            ) / np.sqrt(cluster_sizes[cluster1] + cluster_sizes[cluster2] + cluster_sizes[i])
            new_distances.append(new_distance)
        
        new_distance_matrix = np.vstack((new_distance_matrix, new_distances))
        new_distances.append(0)
        new_distance_matrix = np.column_stack((new_distance_matrix, new_distances))

        return new_distance_matrix

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        if isinstance(model, cls):
            print(f"Model loaded from {filename}")
            return model
        else:
            raise TypeError(f"Expected object of type {cls.__name__}, got {type(model).__name__}")