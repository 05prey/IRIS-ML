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
x_train = pd.read_csv('x_train.csv').drop(columns=['Unnamed: 0']).to_numpy()
x_val = pd.read_csv('x_val.csv').drop(columns=['Unnamed: 0']).to_numpy()
x_test = pd.read_csv('x_test.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_train = pd.read_csv('y_train.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_val = pd.read_csv('y_val.csv').drop(columns=['Unnamed: 0']).to_numpy()
y_test = pd.read_csv('y_test.csv').drop(columns=['Unnamed: 0']).to_numpy()


def rbf_transform(x, centers, r):
    """Apply RBF transformation to the data """
    x_expanded = np.expand_dims(x, 1)
    centers_expanded = np.expand_dims(centers, 0)
    return np.exp(-np.linalg.norm(x_expanded - centers_expanded, axis=2)**2 / r)


def find_rbf_centers(x, k):
    """Find RBF centers using k-means clustering """
    kmeans = KMeans(n_clusters=k, random_state=42).fit(x)
    return kmeans.cluster_centers_


def logistic_regression_with_rbf(x_train, y_train, x_val, y_val, k, r_space):
    """Fit and evaluate logistic regression with RBF transformation """
    centers = find_rbf_centers(x_train, k)
    best_r = None
    best_accuracy = 0

    for r in r_space:
        # uncomment here for contour plotting
        # if k == 10:
            # plt.figure(figsize=(8, 6))
            # plot_rbf_functions(centers, r, "RBF Functions for k={}, r={}".format(k,r))
            # plt.show()
        # Transform features with modified RBF
        x_train_rbf = rbf_transform(x_train, centers, r)
        x_val_rbf = rbf_transform(x_val, centers, r)

        # Fit logistic regression
        dummy_model = LogisticRegression().fit(x_train_rbf, y_train)

        # Evaluate on validation data
        val_accuracy = dummy_model.score(x_val_rbf, y_val)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_r = r

    return best_r, best_accuracy

def plot_rbf_functions(centers, r, title):

    """

    Plot the radial basis functions for given centers and gamma.

    """

    # Create a grid for plotting
    x = np.linspace(-2, 7, 100)
    y = np.linspace(-2, 4, 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Compute RBF responses for each center
    for center in centers:
        rbf_response = np.exp(-np.linalg.norm(grid - center[-2], axis=1)**2 / r)
        zz = rbf_response.reshape(xx.shape)
        plt.contourf(xx, yy, zz, alpha=0.8, levels=25)

    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Centers')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()


# Define r range for tune searching
r_space = 1 / np.logspace(-2, 2, 20)
k_space = [5, 10, 50]

# Recalculate for k = 5, 10, 50 using modified RBF expression
summary= {}
conf_matices = {}
for k in k_space:
    r, val_accuracy = logistic_regression_with_rbf(x_train, y_train, x_val, y_val, k, r_space)
    centers = find_rbf_centers(x_train, k)
    x_train_rbf = rbf_transform(x_train, centers, r)
    x_test_rbf = rbf_transform(x_test, centers, r)

    # Train and evaluate on test data
    best_model = LogisticRegression().fit(x_train_rbf, y_train)
    y_pred_train = best_model.predict(x_train_rbf)
    y_pred_test = best_model.predict(x_test_rbf)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)

    # plt.figure(figsize=(8, 6))
    # plot_rbf_functions(centers, r, "RBF Functions for k={}".format(k))
    # plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=train_conf_matrix, display_labels=best_model.classes_)
    disp.plot()
    plt.title('k={}, confusion matrix of training set'.format(k))
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=test_conf_matrix, display_labels=best_model.classes_)
    disp.plot()
    plt.title('k={}, confusion matrix of test set'.format(k))
    plt.show()

    summary['k='+str(k)] = {
        'best r': r,
        'best val accuracy': val_accuracy,
        'train accuracy': train_accuracy,
        'test accuracy': test_accuracy,
    }
    conf_matices['k='+str(k)] = {
        'model': best_model,
        'train_confusion_matrix': train_conf_matrix,
        'test_confusion_matrix': test_conf_matrix
    }


results = pd.DataFrame(summary, columns=summary.keys())


print(results)






# Plot RBF functions for k = 5

