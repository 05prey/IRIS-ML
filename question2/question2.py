import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
from data.data_loader import DataLoader
from misc.plotter import Plotter

plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)



if __name__ == "__main__":
    df = pd.read_csv('iris.csv')
    df = shuffle(df)
    # drop the Id column, its metadata
    df = df.drop(columns=['Id'])
    print(df.head())

    sns.countplot(df["Species"])
    plt.xlabel('sample count')
    plt.ylabel('species (classes)')
    plt.title('Dataset sample distribution over classes')
    plt.show()

    sns.pairplot(df, hue="Species")
    # plt.title('Pairplot of 4 features')
    plt.show()

    # Boxplots to visualize distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.boxplot(x="Species", y="SepalLengthCm", data=df)
    plt.subplot(2, 2, 2)
    sns.boxplot(x="Species", y="SepalWidthCm", data=df)
    plt.subplot(2, 2, 3)
    sns.boxplot(x="Species", y="PetalLengthCm", data=df)
    plt.subplot(2, 2, 4)
    sns.boxplot(x="Species", y="PetalWidthCm", data=df)
    plt.tight_layout()

    # use preprocessed fixed dataset
    data_loader = DataLoader()
    data_loader.load_data()
    x_train, x_val,x_test, y_train, y_val, y_test = data_loader.get_data()

    # Initialize best accuracy as 0 and best C as None
    best_accuracy = 0
    best_C = None

    C_space = [0.001, 0.01, 0.1, 1, 10, 100]
    # Loop over various values of C
    for C in C_space:
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        accuracy_val = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy for C={C}: {accuracy_val}")

        # Update best C based on validation accuracy
        if accuracy_val > best_accuracy:
            best_accuracy = accuracy_val
            best_C = C

    # Train the final model with the best C value
    print(f"Best C Value: {best_C}")
    model = LogisticRegression(C=best_C, max_iter=1000)
    model.fit(x_train, y_train)

    # Make predictions and evaluate the model
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)

    # Output results
    plotter = Plotter()
    plotter.cm_plotter(cm=conf_matrix_train, model=model, text='confusion matrix of training set')
    plotter.cm_plotter(cm=conf_matrix_test, model=model, text='confusion matrix of test set')

    print("Training Accuracy:{} | Test Accuracy:{}".format(accuracy_train, accuracy_test))
    print(classification_report(y_test, y_test_pred))

