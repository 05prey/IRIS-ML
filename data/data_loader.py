# load same fixed data
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)



class DataLoader():
    def __init__(self):
        self.file_names = ['x_train.csv', 'x_val.csv', 'x_test.csv',
                           'y_train.csv', 'y_val.csv', 'y_test.csv']
        self.data = ()

    def load_data(self):
        for file_name in self.file_names:
            self.data = self.data + (pd.read_csv(file_name).drop(columns=['Unnamed: 0']).to_numpy(),)

    def get_data(self):
        print('getting x_train, x_val, x_test, y_train, y_val, y_test...')
        return self.data


