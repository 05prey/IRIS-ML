from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# use preprocessed fixed dataset
x_train = pd.read_csv('x_train.csv').drop(columns=['Unnamed: 0', '2', '3']).to_numpy()
x_val = pd.read_csv('x_val.csv').drop(columns=['Unnamed: 0', '2', '3']).to_numpy()
x_test = pd.read_csv('x_test.csv').drop(columns=['Unnamed: 0', '2', '3']).to_numpy()
y_train = pd.read_csv('y_train.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_val = pd.read_csv('y_val.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_test = pd.read_csv('y_test.csv').drop(columns=['Unnamed: 0']).to_numpy()


import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def initialize_centroids(self, X):
        np.random.seed(42)  # random seed for randomness
        random_indices = np.random.permutation(X.shape[0])
        centroids = X[random_indices[:self.k]]
        return centroids

    def assign_clusters(self, X, centroids):
        # Assign each point to the closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(self, X, labels):
        # Update centroid location
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return centroids

    def compute_inertia(self, X, labels, centroids):
        # Calculate the inertia (within-cluster sum-of-squares)
        inertia = sum(((X[labels == i] - centroids[i])**2).sum() for i in range(self.k))
        return inertia

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            self.labels = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, self.labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids
        self.inertia = self.compute_inertia(X, self.labels, self.centroids)

X = np.concatenate((x_train, x_val, x_test))

# Run K-Means for k = 2, 5, 10 with different initializations
k_values = [2, 5, 10]
results = {}

for k in k_values:
    best_inertia = None
    best_model = None

    for _ in range(5):  # 5 different initializations
        model = KMeans(k)
        model.fit(X)

        if best_inertia is None or model.inertia < best_inertia:
            best_inertia = model.inertia
            best_model = model

    results[k] = best_model

# Plotting
plt.figure(figsize=(15, 5))

for i, (k, model) in enumerate(results.items(), 1):
    plt.subplot(1, len(k_values), i)
    plt.scatter(X[:, 0], X[:, 1], c=model.labels, cmap='viridis', s=50)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', s=100, marker='x')
    plt.title(f'k = {k}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

