import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_dim, output_dim=20, dropout=0.5):
        super(DNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.fc2(x)
#         x = F.softmax(self.fc2(x), dim=1)
        return x

def build_random_dnn(input_dim, min_hidden_layer, max_hidden_layer, min_nodes, max_nodes, output_dim, dropout):

    layer = list(range(min_hidden_layer,max_hidden_layer))
    node = list(range(min_nodes, max_nodes))

    num_layers = np.random.choice(layer)
    num_nodes = np.random.choice(node)

    dnn = DNN(input_dim, num_layers, num_nodes, output_dim, dropout)

    return dnn