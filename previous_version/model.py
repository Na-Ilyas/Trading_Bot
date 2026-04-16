import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeatureGraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x shape: (Batch, Seq_Len, Features)
        # adj shape: (Features, Features)
        
        # Transform node (feature) states
        support = self.linear(x) # (Batch, Seq_Len, Out_Features)
        
        # Aggregate across feature correlations
        output = torch.matmul(support, adj) 
        return F.relu(output)

class HybridGNNBiLSTM(nn.Module):
    def __init__(self, num_features, seq_length, hidden_dim=64):
        super(HybridGNNBiLSTM, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        
        # Create a learnable adjacency matrix mapping relationships between features
        # (e.g. how RSI interacts with Log Returns)
        self.adj = nn.Parameter(torch.eye(num_features) + torch.randn(num_features, num_features) * 0.1)

        # Layers
        self.gcn1 = FeatureGraphConvLayer(num_features, num_features)
        
        self.bilstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1) 

    def forward(self, x):
        # Normalize adjacency matrix dynamically
        adj_norm = F.softmax(self.adj, dim=1)
        
        # 1. Spatial/Feature Graph Convolution
        x_gcn = self.gcn1(x, adj_norm)
        x_gcn = self.dropout(x_gcn)
        
        # 2. Temporal Sequence Modeling
        lstm_out, _ = self.bilstm(x_gcn)
        
        # Extract last time step
        last_hidden = lstm_out[:, -1, :] 
        
        # 3. Final Classification Logit
        logits = self.fc(last_hidden)
        return logits