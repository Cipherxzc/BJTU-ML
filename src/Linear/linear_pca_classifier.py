import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from safetensors.numpy import save_file, load_file
import pickle


class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.pca = PCA(n_components=50)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 防止溢出
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def fit(self, X, y, num_classes=10):
        X = self.pca.fit_transform(X)
        
        m, n = X.shape
        self.weights = np.zeros((num_classes, n))  # 每个类别一个权重向量
        self.bias = np.zeros(num_classes)  # 每个类别一个偏置
        
        # 将标签转化为one-hot编码
        y_one_hot = np.eye(num_classes)[y]
        
        for _ in tqdm(range(self.iterations), desc="Training Progress", unit="iteration"):
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
        X = self.pca.transform(X)
        Z = np.dot(X, self.weights.T) + self.bias
        predictions = self.softmax(Z)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        X = self.pca.transform(X)
        Z = np.dot(X, self.weights.T) + self.bias
        return self.softmax(Z)
    
    def save_model(self, filename):
        data = {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': np.array([self.learning_rate]),
            'iterations': np.array([self.iterations])
        }
        save_file(data, filename + '_model.safetensors')
        
        with open(filename + '_pca.pkl', 'wb') as pca_file:
            pickle.dump(self.pca, pca_file)
        
        print(f"Model saved to {filename}_model.safetensors and {filename}_pca.pkl")

    def load_model(self, filename):
        data = load_file(filename + '_model.safetensors')
        self.weights = data['weights']
        self.bias = data['bias']
        self.learning_rate = data['learning_rate'][0]
        self.iterations = data['iterations'][0]
        
        with open(filename + '_pca.pkl', 'rb') as pca_file:
            self.pca = pickle.load(pca_file)
        
        print(f"Model loaded from {filename}_model.safetensors and {filename}_pca.pkl")