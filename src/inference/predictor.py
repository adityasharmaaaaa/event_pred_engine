import torch
import pandas as pd
import numpy as np
from src.models.event_transformer import EventPredictor
from src.utils.geo_utils import GeoGrid
import os

class RiskEngine:
    def __init__(self, model_path="outputs/model_v1.pth", load_weights=True):
        self.device = torch.device("cpu")
        
        # Initialize the NEW V2 Model Architecture
        # matching the parameters in your train.py
        self.model = EventPredictor(d_model=128, n_layers=3, dropout=0.2).to(self.device)
        
        self.geo = GeoGrid()
        
        # Only load weights if requested and file exists
        if load_weights and os.path.exists(model_path):
            print(f"✅ Loading model from {model_path}...")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"⚠️ Warning: Could not load weights (Architecture mismatch?): {e}")
                print("   Starting with initialized weights (Fresh Model).")
        else:
            print("🆕 Initializing fresh model (No weights loaded).")

    def preprocess_window(self, df_window):
        """Converts a 7-day DataFrame window into tensors."""
        # Ensure sorted by date
        df_window = df_window.sort_values('Day')
        
        # Extract features
        text_embs = np.stack(df_window['embedding'].values)
        econ_vals = df_window['volatility_7d'].values.reshape(-1, 1)
        
        # Convert to Tensor & Add Batch Dimension (1, 7, Dim)
        txt_tensor = torch.FloatTensor(text_embs).unsqueeze(0).to(self.device)
        eco_tensor = torch.FloatTensor(econ_vals).unsqueeze(0).to(self.device)
        
        return txt_tensor, eco_tensor

    def predict(self, df, hex_id):
        """Predicts risk for a single hex ID given the full dataframe."""
        # Filter for specific hex
        df_loc = df[df['h3_hex'] == hex_id].sort_values('Day')
        
        # We need the LAST 7 days to predict "Tomorrow"
        if len(df_loc) < 7:
            return 0.0 # Not enough data
            
        recent_window = df_loc.tail(7)
        
        txt, eco = self.preprocess_window(recent_window)
        
        self.model.eval()
        with torch.no_grad():
            risk_score = self.model(txt, eco).item()
            
        return risk_score