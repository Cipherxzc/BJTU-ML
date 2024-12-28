import numpy as np
import pickle
from tqdm import tqdm

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None  # 均值
        self.var = None  # 方差
        self.priors = None  # 先验概率
        self.eps = 1e-6

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in tqdm(enumerate(self.classes), total=n_classes, desc="Fitting classes"):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):  # x 属于 class_idx 的概率
        mean = self.mean[class_idx]
        var = self.var[class_idx] + self.eps
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator + self.eps

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx] + self.eps)  # 防止log(0)
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict_proba(self, X):
        proba = [self._predict_proba(x) for x in X]
        return np.array(proba)

    def _predict_proba(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx] + self.eps)  # 防止log(0)
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        max_posterior = max(posteriors)
        exp_posteriors = np.exp(posteriors - max_posterior)
        return exp_posteriors / np.sum(exp_posteriors)

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