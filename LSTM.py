import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=0.5, bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)


        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)
    
        return F.log_softmax(out)


def build_random_lstm(min_hidden_layer_lstm, max_hidden_layer_lstm, min_nodes_lstm, max_nodes_lstm, input_size, output_size):

    values = list(range(min_nodes_lstm,max_nodes_lstm))
    values_layer = list(range(min_hidden_layer_lstm,max_hidden_layer_lstm))

    hidden_dim = np.random.choice(values_layer)
    hidden_layers = np.random.choice(values)
    
    model = LSTM(input_size, output_size, hidden_dim, hidden_layers)


    return model
