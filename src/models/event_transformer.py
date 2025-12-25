import torch
import torch.nn as nn

class EventPredictor(nn.Module):
    def __init__(self, text_dim=384, econ_dim=1, d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        
        # 1. Feature Projectors (Deepened)
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.econ_fc = nn.Sequential(
            nn.Linear(econ_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion
        self.fusion_norm = nn.LayerNorm(d_model * 2)
        
        # 2. Deep Transformer Stack
        # We use a proper TransformerEncoder for the first N-1 layers
        # This allows the model to "reason" deeply before the final attention check
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2, 
            nhead=n_heads, 
            dim_feedforward=512, 
            dropout=dropout,
            batch_first=True
        )
        self.deep_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers - 1)
        
        # 3. Final Interpretable Attention Layer
        # We keep this manual so we can extract weights for the "Red Bar" plot
        self.final_attention = nn.MultiheadAttention(
            embed_dim=d_model * 2, 
            num_heads=n_heads, 
            batch_first=True
        )
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_seq, econ_seq, return_attention=False):
        # A. Project & Fuse
        t_emb = self.text_fc(text_seq) 
        e_emb = self.econ_fc(econ_seq)
        
        x = torch.cat([t_emb, e_emb], dim=2) # (B, W, 256)
        x = self.fusion_norm(x)
        
        # B. Deep Processing (The "Reasoning" Phase)
        x = self.deep_encoder(x)
        
        # C. Final Check (The "Decision" Phase)
        # We use the processed X as Query, Key, and Value
        attn_output, attn_weights = self.final_attention(x, x, x)
        
        # D. Predict based on the last time step
        last_state = attn_output[:, -1, :] 
        prediction = self.classifier(last_state)
        
        if return_attention:
            return prediction, attn_weights
            
        return prediction