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
        x = x + self.pe[:x.size(0), :]
        return x


class AttentionPooling(nn.Module):
    """Learned attention pooling over temporal dimension."""
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: [Batch, Seq_Len, Hidden_Dim]
        attn_weights = self.attention(x)  # [Batch, Seq_Len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)  # [Batch, Hidden_Dim]
        return pooled


class HybridASLModel(nn.Module):
    def __init__(self, input_dim=459, hidden_dim=128, num_classes=100, nhead=4, num_layers=2, dropout=0.5):
        super(HybridASLModel, self).__init__()
        
        # Input normalization
        self.layer_norm_input = nn.LayerNorm(input_dim)
        
        # Linear projection of input features
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder (2 layers, 4 heads — smaller to avoid overfitting)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0
        )
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(hidden_dim * 2)
        
        # FC Head - simpler to reduce overfitting
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        
        # 1. Normalize input
        x = self.layer_norm_input(x)
        
        # 2. Linear projection
        x = self.embedding(x)  # [Batch, Seq_Len, Hidden_Dim]
        
        # 3. Reshape for Transformer [Seq_Len, Batch, Hidden_Dim]
        x = x.permute(1, 0, 2)
        x = self.pos_encoding(x)
        
        # 4. Transformer Encoder
        x = self.transformer_encoder(x)  # [Seq_Len, Batch, Hidden_Dim]
        
        # 5. Reshape for LSTM [Batch, Seq_Len, Hidden_Dim]
        x = x.permute(1, 0, 2)
        lstm_out, _ = self.lstm(x)  # [Batch, Seq_Len, Hidden_Dim * 2]
        
        # 6. Attention Pooling (learns which frames are important)
        x = self.attention_pool(lstm_out)  # [Batch, Hidden_Dim * 2]
        
        # 7. Final FC
        out = self.fc(x)  # [Batch, Num_Classes]
        return out


if __name__ == "__main__":
    # Test model
    model = HybridASLModel(input_dim=459, hidden_dim=128, num_classes=50)
    test_input = torch.randn(8, 60, 459)  # Batch, Seq_Len, Features
    output = model(test_input)
    print(f"Output shape: {output.shape}")  # Should be (8, 50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
