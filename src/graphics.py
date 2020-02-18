import numpy as np
import matplotlib.pyplot as plt

train = open('stats\\train_stats.txt', 'r')
validation = open('stats\\valid_stats.txt', 'r')


def get_stats(file):
    lines = file.readlines()

    epoch = []
    loss = []
    accuracy = []

    i = 0
    for line in range(len(lines)):
        data = lines[i]
        data = data.split(',')
        epoch.append(int(data[0]))
        loss.append(float(data[1]))
        accuracy.append(float(data[2][:-1]))
        i += 1

    return epoch, loss, accuracy


iterations_tr, loss_tr, accuracy_tr = get_stats(train)
iterations_val, loss_val, accuracy_val = get_stats(validation)

fig = plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    if i == 0:
        plt.plot(iterations_tr, loss_tr)
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")

    elif i == 1:
        plt.plot(iterations_tr, accuracy_tr)
        plt.xlabel("Iteration")
        plt.ylabel("Train Accuracy")

    elif i == 2:
        plt.plot(iterations_val, loss_val)
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")

    else:
        plt.plot(iterations_val, accuracy_val)
        plt.xlabel("Iteration")
        plt.ylabel("Validation Accuracy")


min_val_loss = np.argmin(loss_val)
print(iterations_val[min_val_loss], loss_val[min_val_loss], accuracy_val[min_val_loss])

max_val_accuracy = np.argmax(accuracy_val)
print(iterations_val[max_val_accuracy], loss_val[max_val_accuracy], accuracy_val[max_val_accuracy])

plt.show()