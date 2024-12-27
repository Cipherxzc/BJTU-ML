import numpy as np
import pandas as pd
from src.NaiveBayes.train_naive_bayes import train_naive_bayes

train_data = pd.read_csv("data/fashion-mnist_train.csv")
test_data = pd.read_csv("data/fashion-mnist_test.csv")

model = train_naive_bayes(train_data, test_data)