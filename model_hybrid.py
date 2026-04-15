import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Seq_Len, Batch, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

class HybridASLModel(nn.Module):
    def __init__(self, input_dim=306, hidden_dim=128, num_classes=300, nhead=4, num_layers=1, dropout=0.5):
        super(HybridASLModel, self).__init__()
        
        # Linear projection of input features
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0)
        
        # FC Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        
        # 1. Linear projection
        x = self.embedding(x) # [Batch, Seq_Len, Hidden_Dim]
        
        # 2. Reshape for Transformer [Seq_Len, Batch, Hidden_Dim]
        x = x.permute(1, 0, 2)
        x = self.pos_encoding(x)
        
        # 3. Transformer Encoder
        x = self.transformer_encoder(x) # [Seq_Len, Batch, Hidden_Dim]
        
        # 4. Reshape for LSTM [Batch, Seq_Len, Hidden_Dim]
        x = x.permute(1, 0, 2)
        lstm_out, _ = self.lstm(x) # [Batch, Seq_Len, Hidden_Dim * 2]
        
        # 5. Global Average Pooling (across time) or last time step
        # Using Global Average Pooling for more robust features
        x = torch.mean(lstm_out, dim=1) # [Batch, Hidden_Dim * 2]
        
        # 6. Final FC
        out = self.fc(x) # [Batch, Num_Classes]
        return out

if __name__ == "__main__":
    # Test model
    model = HybridASLModel(input_dim=306, hidden_dim=128, num_classes=300)
    test_input = torch.randn(8, 60, 306) # Batch, Seq_Len, Features
    output = model(test_input)
    print(f"Output shape: {output.shape}") # Should be (8, 300)
