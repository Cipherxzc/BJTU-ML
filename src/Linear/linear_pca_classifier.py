import numpy as np
import pickle

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 防止溢出
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def fit(self, X, y, num_classes):
        m, n = X.shape
        self.weights = np.zeros((num_classes, n))  # 每个类别一个权重向量
        self.bias = np.zeros(num_classes)  # 每个类别一个偏置
        
        # 将标签转化为one-hot编码
        y_one_hot = np.eye(num_classes)[y]
        
        for _ in range(self.iterations):
            # 计算模型输出
            Z = np.dot(X, self.weights.T) + self.bias
            predictions = self.softmax(Z)
            
            # 计算损失函数的梯度
            dw = (1 / m) * np.dot((predictions - y_one_hot).T, X)  # 对权重求导
            db = (1 / m) * np.sum(predictions - y_one_hot, axis=0)  # 对偏置求导
            
            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        Z = np.dot(X, self.weights.T) + self.bias
        predictions = self.softmax(Z)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        Z = np.dot(X, self.weights.T) + self.bias
        return self.softmax(Z)
    
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