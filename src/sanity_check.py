import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, PROCESSED_DIR, ONE_MIN_PARQUET_DIR
from src.models.cnn_model import TradeCNN

def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradeCNN().to(device)
    path = MODELS_DIR / "rejection_cnn_v1.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {path}")
    
    # Check Bias
    zeros = torch.zeros(1, 4, 20).to(device)
    with torch.no_grad():
        out = model(zeros).item()
    print(f"Model(Zeros): {out:.4f}")
    
    # Reproduce One Sample from Strategy Logic
    trades = pd.read_parquet(PROCESSED_DIR / "labeled_rejections.parquet").sort_values('start_time')
    
    # Pick a random trade from the end
    trade = trades.iloc[-100]
    print(f"Checking Trade: {trade['start_time']} ({trade['direction']}) Winner? {trade['outcome']}")
    
    df_1m = pd.read_parquet(ONE_MIN_PARQUET_DIR / "mes_1min_all.parquet")
    df_1m['time'] = pd.to_datetime(df_1m['time'], utc=True)
    df_1m = df_1m.set_index('time').sort_index()
    
    end_time = trade['start_time']
    start_time = end_time - pd.Timedelta(minutes=20)
    window = df_1m.loc[start_time:end_time]
    window = window[window.index < end_time]
    
    if len(window) < 20:
        print("Window too short")
        return

    feats = window[['open', 'high', 'low', 'close']].values[-20:]
    
    # Z-Score
    mean = np.mean(feats)
    std = np.std(feats)
    if std == 0: std = 1.0
    feats_norm = (feats - mean) / std
    
    if trade['direction'] == 'SHORT':
        feats_norm = -feats_norm
        
    inp = torch.FloatTensor(feats_norm).unsqueeze(0).to(device)
    if inp.shape[1] != 4: inp = inp.permute(0, 2, 1)
        
    with torch.no_grad():
        p = model(inp).item()
        
    print(f"Prediction for Sample: {p:.4f}")

if __name__ == "__main__":
    check()
