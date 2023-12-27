import pandas as pd
from data.data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)



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
        # assign each data point to the closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(self, X, labels):
        # update centroid location
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return centroids

    def compute_Ein(self, X, labels, centroids):
        # calculate the error measure, sum of square errors (within-cluster sum-of-squares)
        self.error = sum(((X[labels == i] - centroids[i])**2).sum() for i in range(self.k))

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            self.labels = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, self.labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids
        self.compute_Ein(X, self.labels, self.centroids)



if __name__ == "__main__":

    # use preprocessed fixed dataset
    data_loader = DataLoader()
    data_loader.load_data()
    x_train, x_val,x_test, y_train, y_val, y_test = data_loader.get_data()

    # use whole data for clustering
    X = np.concatenate((x_train, x_val, x_test))

    # run K-Means for k = 2, 5, 10 with different initializations
    k_space = [2, 5, 10]
    results = {}

    for k in k_space:
        best_error = None
        best_model = None

        for _ in range(5):  # 5 different initializations
            model = KMeans(k)
            model.fit(X)

            if best_error is None or model.error < best_error:
                best_error = model.error
                best_model = model

        results[k] = best_model

    # plotting
    plt.figure(figsize=(15, 5))

    for i, (k, model) in enumerate(results.items(), 1):
        plt.subplot(1, len(k_space), i)
        plt.scatter(X[:, 0], X[:, 1], c=model.labels, cmap='viridis', s=50)
        plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', s=100, marker='x')
        plt.title(f'k = {k}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

