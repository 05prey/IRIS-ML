import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils import shuffle

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

    # split features and labels
    x = df.drop(columns=['Species'])
    y = df['Species']

    # split the data
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # standardization (must be done seperately in order to prevent data leakage)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Initialize best accuracy as 0 and best C as None
    best_accuracy = 0
    best_C = None

    # Loop over various values of C
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
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
    print("Training Accuracy:", accuracy_train)
    print("Test Accuracy:", accuracy_test)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_train, display_labels=model.classes_)
    disp.plot()
    plt.title('confusion matrix of training set')
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=model.classes_)
    disp.plot()
    plt.title('confusion matrix of test set')
    plt.show()

    print(classification_report(y_test, y_test_pred))

