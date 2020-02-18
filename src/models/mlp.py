import torch
import torch.nn as nn
import data_loader
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 13
hidden_size = 256
num_classes = 2
num_epochs = 1000
batch_size = 16
learning_rate = 0.001

# Dataset
train_x = 'x_train.pkl'
train_y = 'y_train.pkl'

valid_x = 'x_valid.pkl'
valid_y = 'y_valid.pkl'

test_x = 'x_test.pkl'
test_y = 'y_test.pkl'

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=data_loader.data_loader(train_x, train_y),
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=data_loader.data_loader(valid_x, valid_y),
                                          batch_size=batch_size,
                                          shuffle=True)
#
test_loader = torch.utils.data.DataLoader(dataset=data_loader.data_loader(test_x, test_y),
                                          batch_size=batch_size,
                                          shuffle=True)


# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


# Model
model = MLP(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

# Statisctics
train_stats = open('stats/train_stats.txt', 'w')
valid_stats = open('stats/valid_stats.txt', 'w')


# Train the model
total_train = 0
correct_train = 0
iteration = 0
current_loss = 0
train_it = 500
valid_it = 200

total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (data, labels) in enumerate(train_loader):
        iteration += 1
        # Forward pass
        outputs = model(data.float())
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Loss
        current_loss += loss.item()

        if iteration % train_it == 0:
            train_stats.write(('{},{:.4f},{:.2f}'
                               .format(iteration, current_loss / train_it,
                                       (correct_train / total_train) * 100)) + '\n')
            print('Iteration: {}, Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(iteration, current_loss / train_it, (correct_train / total_train) * 100))
            correct_train = 0
            total_train = 0
            current_loss = 0


        if iteration % valid_it == 0:
            model.eval()
            with torch.no_grad():
                correct_val = 0
                total_val = 0
                valid_loss = 0
                total_val_steps = len(validation_loader)
                for data, labels in validation_loader:
                    outputs = model(data.float())
                    labels = labels.to(device)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                # Save the model checkpoint
                torch.save(model.state_dict(), 'MLP_models\\model{}.ckpt'.format(iteration))
                valid_stats.write(('{},{:.4f},{:.2f}'
                     .format(iteration, valid_loss / total_val_steps, (correct_val / total_val) * 100)) + '\n')
                print('Iteration: {}, Loss: {:.4f}, Valid Accuracy: {:.2f}%'
                     .format(iteration, valid_loss / total_val_steps, (correct_val / total_val) * 100))


print('Training is finished!')
train_stats.close()
valid_stats.close()


# Test the model

# Confusion matrix
def confusion_matrix(true_labels, predicted_labels):
    size = (num_classes, num_classes)
    cm = np.zeros(size, dtype=int)

    for i in range(len(true_labels)):
        cm[true_labels[i], predicted_labels[i]] += 1

    return cm


# Loading model parametrs
checkpoint = torch.load('stats\\model800.ckpt')
model.load_state_dict(checkpoint)

cm = np.zeros((num_classes, num_classes), dtype=int)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        labels = labels.to(device)
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        cm += confusion_matrix(labels, predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the MLP on test set: {} %'.format(100 * correct / total))
    print(cm)


# Plotting a confusion matrix
df_cm = pd.DataFrame(cm, index=[i for i in ['Healthy', 'Heart disease']], columns=[i for i in ['Healthy', 'Heart disease']])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
