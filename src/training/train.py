import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from src.models.event_transformer import EventPredictor
from src.inference.predictor import RiskEngine
import os

def train_model():
    print("🚀 Training V2: Deep Transformer...")
    
    # 1. Load Data
    data_path = "data/processed/training_sets/train_multimodal.parquet"
    if not os.path.exists(data_path):
        print("❌ Data not found.")
        return
        
    df = pd.read_parquet(data_path)
    engine = RiskEngine(load_weights=False) # Helper to format data
    
    # 2. Build Sequences
    print("Building temporal sequences...")
    X_text, X_econ, y = [], [], []
    
    # Group by Hex to respect temporal order
    for _, group in df.groupby('h3_hex'):
        group = group.sort_values('Day')
        if len(group) < 7: continue
            
        # Sliding window
        t_emb = np.stack(group['embedding'].values)
        e_val = group['volatility_7d'].values.reshape(-1, 1)
        targets = group['target_label'].values
        
        for i in range(len(group) - 7):
            X_text.append(t_emb[i:i+7])
            X_econ.append(e_val[i:i+7])
            y.append(targets[i+7]) # Predict next day
            
    # Convert to Tensors
    X_text = torch.FloatTensor(np.array(X_text))
    X_econ = torch.FloatTensor(np.array(X_econ))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)
    
    dataset = TensorDataset(X_text, X_econ, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Initialize Model (Bigger & Deeper)
    device = torch.device("cpu") # M1 Mac supports mps but cpu is stable for small batch
    model = EventPredictor(d_model=128, n_layers=3, dropout=0.2).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.BCELoss()
    
    # 4. Training Loop
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_txt, batch_eco, batch_y in loader:
            batch_txt, batch_eco, batch_y = batch_txt.to(device), batch_eco.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_txt, batch_eco)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        
        # Step the scheduler
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "outputs/model_v1.pth") # Overwrite V1
            
    print(f"✅ Model trained. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_model()