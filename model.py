import torch.nn as nn
from copy import deepcopy

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPModel, self).__init__()

        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        x = x.view(-1, 8, 8)
        out, _ = self.lstm(x)
        x = self.fc(out[:, -1, :])
        return x

    def get_copy(self):
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(deepcopy(state_dict))

def get_model(model_name):
    models = {
        'MLP': MLPModel(input_size=64, hidden_sizes=[128, 64, 32], output_size=4),
        'NN': NNModel(input_size=64, hidden_size=64, output_size=4),
        'LSTM': LSTMModel()
    }

    return models[model_name]