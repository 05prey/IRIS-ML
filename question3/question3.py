from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix
import math

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)



class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.classes_ = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for point in X:
            # Compute distances from the point to all training samples
            distances = np.linalg.norm(self.x_train - point, axis=1)
            # Get the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Get the labels of the k nearest neighbors
            k_labels = self.y_train[k_indices]
            # Predict the label by majority vote
            pred_label = mode(k_labels).mode[0]
            predictions.append(pred_label)
        return np.array(predictions)


# use preprocessed fixed dataset
x_train = pd.read_csv('x_train.csv').drop(columns=['Unnamed: 0']).to_numpy()
x_val = pd.read_csv('x_val.csv').drop(columns=['Unnamed: 0']).to_numpy()
x_test = pd.read_csv('x_test.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_train = pd.read_csv('y_train.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_val = pd.read_csv('y_val.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_test = pd.read_csv('y_test.csv').drop(columns=['Unnamed: 0']).to_numpy()

x_train_val = np.concatenate((x_train, x_val))
y_train_val = np.concatenate((y_train, y_val))






# k values
k1 = 3
k2 = math.ceil(math.sqrt(len(x_train))) # k = sqrt(N)
k_values = [k1, k2]

# Perform cross-validation to find the optimal k
def cross_validation_knn(x, y, k_values, num_folds=5):
    fold_size = len(x) // num_folds
    best_k = k_values[0]
    best_accuracy = 0

    for k in k_values:
        accuracies = []
        for fold in range(num_folds):
            start, end = fold * fold_size, (fold + 1) * fold_size
            x_val_fold = x[start:end]
            y_val_fold = y[start:end]
            x_train_fold = np.concatenate([x[:start], x[end:]])
            y_train_fold = np.concatenate([y[:start], y[end:]])

            model = KNNClassifier(k)
            model.fit(x_train_fold, y_train_fold)
            predictions = model.predict(x_val_fold)
            accuracy = accuracy_score(y_val_fold, predictions)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_k = k

    return best_k

# Find optimal k using cross-validation
k3 = cross_validation_knn(x_train_val, y_train_val, range(1, 20))
k_values.append(k3)

# Evaluate each classifier
results = {}
for i, k in enumerate(k_values):
    knn = KNNClassifier(k)
    knn.fit(x_train, y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)

    disp = ConfusionMatrixDisplay(confusion_matrix=train_conf_matrix, display_labels=knn.classes_)
    disp.plot()
    plt.title('k_{}={}, confusion matrix of training set'.format(i+1,k))
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=test_conf_matrix, display_labels=knn.classes_)
    disp.plot()
    plt.title('k_{}={}, confusion matrix of test set'.format(i+1,k))
    plt.show()

    results['k_{}='.format(i+1)+str(k)] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        # 'train_confusion_matrix': train_conf_matrix,
        # 'test_confusion_matrix': test_conf_matrix
    }

results = pd.DataFrame(results, columns=results.keys())

print(results)
