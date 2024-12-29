import numpy as np
import pickle
from tqdm import tqdm

class ShallowDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, sample_weight=None):
        self.tree = self._build_tree(X, y, sample_weight, depth=0)

    def _build_tree(self, X, y, sample_weight, depth):
        # 停止条件：到达最大深度或样本不可分
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) == 0:
            if len(y) == 0:  # 如果没有样本
                return 0  # 返回默认类别
            counts = np.bincount(y, weights=sample_weight) if sample_weight is not None else np.bincount(y)
            return np.argmax(counts)

        # 找到最佳分裂点
        best_feature, best_threshold = self._find_best_split(X, y, sample_weight)

        if best_feature is None:
            # 如果无法找到分裂点，返回当前节点最常见的类别
            return np.argmax(np.bincount(y, weights=sample_weight) if sample_weight is not None else np.bincount(y))

        # 划分数据
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        left_child = self._build_tree(X[left_indices], y[left_indices], sample_weight[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], sample_weight[right_indices], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_child, "right": right_child}

    def _find_best_split(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_impurity = float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                if sample_weight is not None:
                    impurity = self._weighted_gini(y, left_indices, right_indices, sample_weight)
                else:
                    impurity = self._weighted_gini(y, left_indices, right_indices, np.ones(len(y)))

                if impurity < best_impurity:
                    best_feature = feature
                    best_threshold = threshold
                    best_impurity = impurity

        return best_feature, best_threshold

    def _weighted_gini(self, y, left_indices, right_indices, sample_weight):
        left_weight = np.sum(sample_weight[left_indices])
        right_weight = np.sum(sample_weight[right_indices])

        total_weight = left_weight + right_weight
        if total_weight == 0:
            return 0

        left_gini = 1 - np.sum((np.bincount(y[left_indices], weights=sample_weight[left_indices]) / left_weight) ** 2)
        right_gini = 1 - np.sum((np.bincount(y[right_indices], weights=sample_weight[right_indices]) / right_weight) ** 2)

        return (left_weight * left_gini + right_weight * right_gini) / total_weight

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._traverse_tree(self.tree, sample))
        return np.array(predictions)

    def _traverse_tree(self, node, sample):
        if isinstance(node, dict):  # 确保当前节点是字典
            if "feature" in node and "threshold" in node:
                if sample[node["feature"]] <= node["threshold"]:
                    return self._traverse_tree(node["left"], sample)
                else:
                    return self._traverse_tree(node["right"], sample)
            else:
                raise ValueError("Tree node is missing 'feature' or 'threshold'.")
        else:
            return node

class AdaBoost:
    def __init__(self, base_model, n_estimators=50, **base_model_params):
        self.base_model = base_model
        self.base_model_params = base_model_params  # 决策树的深度
        self.n_estimators = n_estimators
        self.models = []  # 保存每个弱分类器
        self.model_weights = []  # 保存每个弱分类器的权重
        self.classes_ = None
        self.eps = 1e-6

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # 获取所有类别
        n_samples, n_classes = X.shape[0], len(self.classes_)

        # 初始化样本权重
        weights = np.ones(n_samples) / n_samples

        for i in tqdm(range(self.n_estimators), desc="Training AdaBoost", unit="estimator"):
            # 创建一个弱分类器实例
            model = self.base_model(**self.base_model_params)

            # 将样本权重传递给弱分类器进行训练
            model.fit(X, y, sample_weight=weights)

            # 弱分类器预测类别
            y_pred = model.predict(X)

            # 计算分类错误率
            incorrect = (y_pred != y).astype(float)
            error = np.sum(weights * incorrect) / np.sum(weights)
            
            # 测试使用
            print(f"The {i+1} base_model's error: {error:.4f}\n")

            # 如果分类误差为 0 或 >= 0.5，停止训练
            if error >= 0.5-self.eps or error == 0:
                print("Error: error >= 0.5 or error == 0\n")
                break

            # 计算弱分类器的权重
            alpha = 0.5 * np.log((1 - error) / error)

            # 更新样本权重
            weights *= np.exp(alpha * (y_pred != y).astype(float))
            weights = np.clip(weights, 1e-10, 1e10)  # 防止数值溢出或不足
            weights /= np.sum(weights)  # 归一化

            # 保存弱分类器及其权重
            self.models.append(model)
            self.model_weights.append(alpha)

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_scores = np.zeros((n_samples, n_classes))  # 初始化得分矩阵

        # 累加每个弱分类器的加权预测
        for model, alpha in zip(self.models, self.model_weights):
            y_pred = model.predict(X)  # 弱分类器的预测结果
            for i, cls in enumerate(self.classes_):
                class_scores[:, i] += alpha * (y_pred == cls).astype(float)

        # 返回得分最高的类别
        return self.classes_[np.argmax(class_scores, axis=1)]

    def predict_proba(self, X):
        class_scores = np.zeros((X.shape[0], len(self.classes_)))

        for model, alpha in zip(self.models, self.model_weights):
            y_pred = model.predict(X)
            for i, cls in enumerate(self.classes_):
                class_scores[:, i] += alpha * (y_pred == cls).astype(float)

        # 将得分转化为概率
        exp_scores = np.exp(class_scores)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities

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
