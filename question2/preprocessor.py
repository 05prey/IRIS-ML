# this file prepares the dataset, run only once
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    pd.DataFrame(x_train).to_csv('x_train.csv')
    pd.DataFrame(x_val).to_csv('x_val.csv')
    pd.DataFrame(x_test).to_csv('x_test.csv')
    pd.DataFrame(y_train).to_csv('y_train.csv')
    pd.DataFrame(y_val).to_csv('y_val.csv')
    pd.DataFrame(y_test).to_csv('y_test.csv')
