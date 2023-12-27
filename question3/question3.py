import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix
import math
from data.data_loader import DataLoader
from misc.plotter import Plotter

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)



class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.classes_ = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

    def contain(self, X, y):
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


# Perform cross-validation to find the optimal k
def cross_validation(x, y, k_space, num_folds=5):
    fold_size = len(x) // num_folds
    best_k = k_space[0]
    best_accuracy = 0

    for k in k_space:
        accuracies = []
        for fold in range(num_folds):
            start, end = fold * fold_size, (fold + 1) * fold_size
            x_val_fold = x[start:end]
            y_val_fold = y[start:end]
            x_train_fold = np.concatenate([x[:start], x[end:]])
            y_train_fold = np.concatenate([y[:start], y[end:]])

            model = KNNClassifier(k)
            model.contain(x_train_fold, y_train_fold)
            predictions = model.predict(x_val_fold)
            accuracy = accuracy_score(y_val_fold, predictions)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_k = k

    return best_k


if __name__ == "__main__":

    # use preprocessed fixed dataset
    data_loader = DataLoader()
    data_loader.load_data()
    x_train, x_val,x_test, y_train, y_val, y_test = data_loader.get_data()

    x_train_val = np.concatenate((x_train, x_val))
    y_train_val = np.concatenate((y_train, y_val))


    # k values
    k1 = 3
    k2 = math.ceil(math.sqrt(len(x_train))) # k = sqrt(N)
    k_space = [k1, k2]


    plotter = Plotter()

    # Find optimal k using cross-validation
    k3 = cross_validation(x_train_val, y_train_val, range(1, 20))
    k_space.append(k3)

    # Evaluate each classifier
    results = {}
    for i, k in enumerate(k_space):
        knn = KNNClassifier(k)
        knn.contain(x_train, y_train)
        y_pred_train = knn.predict(x_train)
        y_pred_test = knn.predict(x_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_conf_matrix = confusion_matrix(y_train, y_pred_train)
        test_conf_matrix = confusion_matrix(y_test, y_pred_test)

        plotter.cm_plotter(cm=train_conf_matrix, model=knn, text='k_{}={}, confusion matrix of training set'.format(i+1,k))
        plotter.cm_plotter(cm=test_conf_matrix, model=knn, text='k_{}={}, confusion matrix of test set'.format(i+1,k))

        results['k_{}='.format(i+1)+str(k)] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
        }

    results = pd.DataFrame(results, columns=results.keys())

    print(results)
