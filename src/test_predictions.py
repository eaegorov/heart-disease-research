from models.mlp import model
import torch
import numpy as np
from sklearn import preprocessing


def get_prediction(data):
    data = np.array(data).reshape(1, len(data))

    data = preprocessing.normalize(data)
    data = torch.from_numpy(data)

    output = model(data.float())
    _, predicted = torch.max(output.data, 1)
    prediction = int(predicted[0])

    if prediction == 0:
        print('Пациент здоров')
    else:
        print('Имеется заболевание сердца')


# Loading model parametrs
checkpoint = torch.load('stats\\model800.ckpt')
model.load_state_dict(checkpoint)
model.eval()

# Number of features
n = 13

data = []
for i in range(n):
    x = float(input())
    data.append(x)

get_prediction(data)
