import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('heart.csv')  # 303x14 dataset

x_train, x_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size=0.2, random_state=10)
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=10)

x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)

x_train = preprocessing.normalize(x_train)
x_valid = preprocessing.normalize(x_valid)
x_test = preprocessing.normalize(x_test)

with open('x_train.pkl', 'wb') as f:
    pickle.dump(x_train, f)

with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('x_valid.pkl', 'wb') as f:
    pickle.dump(x_valid, f)

with open('y_valid.pkl', 'wb') as f:
    pickle.dump(y_valid, f)

with open('x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)

with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
