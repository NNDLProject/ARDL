import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DNN(nn.Module):
    def __init__(self, input_size, emb_dim, emb_weights, hidden_layers, hidden_dims, output_dim=20, dropout=0.3, max_len=500):
        super(DNN, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_dim)
        
        if emb_weights is not None:
          self.embedding.weight=nn.Parameter(torch.tensor(emb_weights,dtype=torch.float32))
          self.embedding.weight.requires_grad=False
  
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(emb_dim*max_len, hidden_dims[0])
        self.hidden = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(hidden_layers-1)])
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.squeeze(2)
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.fc2(x)
#         x = F.softmax(self.fc2(x), dim=1)
        return F.log_softmax(x)

def build_random_dnn(input_dim, emb_dim, emb_weights, min_hidden_layer, max_hidden_layer, min_nodes, max_nodes, output_dim, dropout=0.3):

    layer = list(range(min_hidden_layer,max_hidden_layer))
    node = list(range(min_nodes, max_nodes))

    num_layers = np.random.choice(layer)
    num_nodes = []
    for layer in range(num_layers):
      num_nodes.append(np.random.choice(node))

    dnn = DNN(input_dim, emb_dim, emb_weights, num_layers, num_nodes, output_dim, dropout)

    return dnn