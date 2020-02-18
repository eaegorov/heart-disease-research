import numpy as np
from matplotlib import pyplot as plt
import torchvision
import pickle
import pandas as pd
import seaborn as sn

K = 2  # Количество классов

# Функция, которая возвращает предсказания
def predict(X, W, B):
    y = softmax(X @ W + B)
    return y


# Функция подсчета accuracy
def accuracy(predicted, real):
    correct = 0
    total = real.shape[0]
    for i in range(total):
        p = predicted[i, :]
        t = real[i, :]
        if (np.argmax(p) == np.argmax(t)):
            correct += 1

    return round((correct / total) * 100, 2)


# Softmax
def softmax(Z):
    for i in range(len(Z)):
        a = np.max(Z[i, :])
        Z[i, :] -= a
        Z[i, :] = np.exp(Z[i, :]) / np.sum(np.exp(Z[i, :]))

    return Z


def E_loss(X, W, B, T):
    Y = predict(X, W, B)
    loss = 0
    for i in range(len(Y)):
        predictions = Y[i, :]
        c = np.argmax(T[i, :])
        loss += -np.log(predictions[c])

    return loss / len(Y)


# Вычисление градиента
def E_gradient(X, W, B, T):
    lambd = 0.0005  # Regularization coefficient
    Y = predict(X, W, B)
    w_grad = (Y - T).T @ X

    U = np.ones((X.shape[0], 1))
    b_grad = (Y - T).T @ U

    return b_grad.T, w_grad.T + lambd * W


# Мини-батчевый градиентный спуск
def gradient_descent(x_train, y_train, x_valid, y_valid, lr):
    # Initialization
    w_next, b_next = initialization(x_train, K)

    best_acc = 0
    best_it = -1
    best_w = 0
    best_b = 0
    eps = 0.000001
    batch_size = 16
    optimize = True
    iteration = 1
    while optimize and iteration <= 1000000:
        # Mini-batch generation
        idxs = np.random.randint(0, len(x_train), size=batch_size)
        batch = np.array([x_train[i, :] for i in idxs]).reshape(batch_size, x_train.shape[1])
        t = np.array([y_train[i, :] for i in idxs]).reshape(batch_size, y_train.shape[1])

        w_old = w_next
        b_old = b_next
        b_grad, w_grad = E_gradient(batch, w_old, b_old, t)

        w_next = w_old - lr * w_grad
        b_next = b_old - lr * b_grad

        if iteration % 100000 == 0:
            train_predictions = predict(x_train, w_next, b_next)
            valid_predictions = predict(x_valid, w_next, b_next)
            train_acc = accuracy(train_predictions, y_train)
            valid_acc = accuracy(valid_predictions, y_valid)
            print('Iteration: {}'.format(iteration))
            print('Train accuracy:', train_acc)
            print('Train loss:', E_loss(x_train, w_next, b_next, y_train))
            print('Validation accuracy:', valid_acc)
            print('Validation loss:', E_loss(x_valid, w_next, b_next, y_valid))
            print('-------------------')
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_it = iteration
                best_w, best_b = w_next, b_next

        norm = np.sqrt(np.sum((w_next - w_old) ** 2))
        if norm < eps:
            break
        iteration += 1

    print('The best accuracy is {} on {} iteration'.format(best_acc, best_it))
    # Saving params
    with open('weights.pkl'.format(iteration), 'wb') as f:
        pickle.dump(best_w, f)
    with open('bias.pkl'.format(iteration), 'wb') as f:
        pickle.dump(best_b, f)

    return best_w, best_b

# Weights and bias initialization
def initialization(x_train, num_classes):
    var = 0.001
    W = np.random.normal(0.0, var, (x_train.shape[1], num_classes))
    B = np.random.normal(0.0, var, (num_classes, 1))
    return W, B.T


# Converting labels to one-hot-encoding vector
def one_hot_encoding(x, y, num_classes):
    y = np.array(y)
    y_one_hot = np.zeros((x.shape[0], num_classes))
    for i in range(len(y)):
        y_one_hot[i, y[i]] = 1

    return y_one_hot


def confusionMaxtrix(predicted, true, num_classes):
    size = (num_classes, num_classes)
    conf_matrix = np.zeros(size, dtype=int)
    total = true.shape[0]

    for i in range(total):
        p = np.argmax(predicted[i, :])
        t = np.argmax(true[i, :])
        conf_matrix[t, p] += 1

    return conf_matrix


# Train mode
def train(x_train, y_train, x_valid, y_valid, learning_rate):
    w, b = gradient_descent(x_train, y_train, x_valid, y_valid, learning_rate)
    print('Train is finished!')

    return w, b


# Eval mode
def eval(x_train, y_train, x_test, y_test):
    weights_name = 'weights.pkl'
    with open(weights_name, 'rb') as f:
        w = pickle.load(f)

    bias_name = 'bias.pkl'
    with open(bias_name, 'rb') as f:
        b = pickle.load(f)

    print('Evaluation:')
    train_predictions = predict(x_train, w, b)
    print('Train accuracy:', accuracy(train_predictions, y_train))
    test_predictions = predict(x_test, w, b)
    print('Test accuracy:', accuracy(test_predictions, y_test))

    return test_predictions


if __name__ == '__main__':
    # Train data
    with open('x_train.pkl', 'rb') as f:
        x_train = pickle.load(f)
    with open('y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    # Validation data
    with open('x_valid.pkl', 'rb') as f:
        x_valid = pickle.load(f)
    with open('y_valid.pkl', 'rb') as f:
        y_valid = pickle.load(f)

    # Test data
    with open('x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    # Converting to one-hot encoding vector
    y_train = one_hot_encoding(x_train, y_train, K)
    y_valid = one_hot_encoding(x_valid, y_valid, K)
    y_test = one_hot_encoding(x_test, y_test, K)

    # Train
    learning_rate = 0.001
    w, b = train(x_train, y_train, x_valid, y_valid, learning_rate=learning_rate)

    # Evaluation
    test_pred = eval(x_train, y_train, x_test, y_test)

    # Confusion Matrix for the TEST SET
    CM = confusionMaxtrix(test_pred, y_test, K)
    print('Confusion matrix:', CM)
    df_cm = pd.DataFrame(CM, index=[i for i in ['Healthy', 'Heart disease']],
                         columns=[i for i in ['Healthy', 'Heart disease']])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
