import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=10, hidden_dim=64, num_layers=2, num_heads=4):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # Embedding: (batch_size, seq_len, hidden_dim)
        x = self.positional_encoding(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # Transpose dimensions for transformer: (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)  # Transformer encoding: (seq_len, batch_size, hidden_dim)
        x = x.mean(dim=0)  # Average pooling over sequence length: (batch_size, hidden_dim)
        x = self.fc(x)  # Fully connected layer: (batch_size, output_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
        self.positional_encoding[:, :, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x

# Define the model
# input_dim = 1024
# output_dim = 10
# hidden_dim = 64
# num_layers = 2
# num_heads = 4
# batch_size = 64
#
# model = TransformerEncoder(input_dim, output_dim, hidden_dim, num_layers, num_heads)
#
# # Create a random input tensor of size (64, 1, 1024)
# input_tensor = torch.randn(batch_size, 1, input_dim)
#
# # Forward pass
# output_tensor = model(input_tensor)
#
# print("Input size:", input_tensor.size())
# print("Output size:", output_tensor.size())
