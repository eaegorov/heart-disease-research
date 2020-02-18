import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np


def unpickle_it(x_train_path, y_train_path, x_valid_path, y_valid_path, x_test_path, y_test_path):
    # Train data
    with open(x_train_path, 'rb') as f:
        x_train = pickle.load(f)
    with open(y_train_path, 'rb') as f:
        y_train = pickle.load(f)

    # Validation data
    with open(x_valid_path, 'rb') as f:
        x_valid = pickle.load(f)
    with open(y_valid_path, 'rb') as f:
        y_valid = pickle.load(f)

    # Test data
    with open(x_test_path, 'rb') as f:
        x_test = pickle.load(f)
    with open(y_test_path, 'rb') as f:
        y_test = pickle.load(f)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def confusion_matrix(true_labels, predicted_labels):
    num_classes = 2
    size = (num_classes, num_classes)
    cm = np.zeros(size, dtype=int)

    for i in range(len(true_labels)):
        cm[true_labels[i], predicted_labels[i]] += 1

    return cm


x_train, y_train, x_valid, y_valid, x_test, y_test = unpickle_it('x_train.pkl', 'y_train.pkl', 'x_valid.pkl', 'y_valid.pkl', 'x_test.pkl', 'y_test.pkl')

rf = RandomForestClassifier(n_estimators=250, random_state=11)
rf.fit(x_train, y_train)

acc_train = np.mean(y_train == rf.predict(x_train))
acc_test = np.mean(y_test == rf.predict(x_test))

print('Train accuracy:', acc_train * 100)
print('Test accuracy:', acc_test * 100)

CM = confusion_matrix(y_test, rf.predict(x_test))

# Confusion matrix visualization
df_cm = pd.DataFrame(CM, index=[i for i in ['Healthy', 'Heart disease']], columns=[i for i in ['Healthy', 'Heart disease']])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Feature importances
importances = rf.feature_importances_

for i in range(len(importances)):
    print(round(importances[i], 4))
