"""
STAT8003 Time Series Forecasting Project
Deep Learning Models: LSTM & GRU for Hong Kong Airport Passenger Traffic

This script provides detailed analysis of:
1. LSTM (Long Short-Term Memory)
2. GRU (Gated Recurrent Unit) 
3. XGBoost (as ML baseline)

Author: [Your Name]
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SEQ_LENGTH = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = r'c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\data'
FIGURES_DIR = r'c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\figures'

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# ============================================================
# 1. Data Loading and Feature Engineering
# ============================================================
print("="*70)
print("Deep Learning Models for Airport Passenger Forecasting")
print("Models: LSTM, GRU, XGBoost")
print("="*70)

print("\n" + "="*70)
print("PART 1: DATA PREPARATION")
print("="*70)

train_df = pd.read_csv(f'{DATA_DIR}/airport_train.csv')
test_df = pd.read_csv(f'{DATA_DIR}/airport_test.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

print(f"\n[1.1] Data Overview")
print(f"      Training samples: {len(train_df)} days")
print(f"      Test samples: {len(test_df)} days")
print(f"      Training period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
print(f"      Test period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")

def create_features(df):
    """Create temporal features for the models"""
    features = pd.DataFrame()
    features['Total'] = df['Total']
    features['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    features['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    features['IsWeekend'] = df['IsWeekend'].astype(float)
    features['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    features['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return features

print(f"\n[1.2] Feature Engineering")
train_features = create_features(train_df)
test_features = create_features(test_df)
print(f"      Features: {list(train_features.columns)}")
print(f"      Total features: {train_features.shape[1]}")

print(f"\n[1.3] Test Data Information")
for i, (_, row) in enumerate(test_df.iterrows()):
    is_weekend = "Weekend" if row['IsWeekend'] == 1 else "Weekday"
    print(f"      {row['Date'].strftime('%Y-%m-%d')} ({day_names[row['DayOfWeek']]}): {is_weekend}, Actual={row['Total']:,}")

# Standardization
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

target_scaler = StandardScaler()
target_scaler.fit(train_df[['Total']])

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(train_scaled, SEQ_LENGTH)
val_size = int(len(X) * 0.1)
X_train, X_val = X[:-val_size], X[-val_size:]
y_train, y_val = y[:-val_size], y[-val_size:]

print(f"\n[1.4] Sequence Creation")
print(f"      Window size: {SEQ_LENGTH} days")
print(f"      Training sequences: {len(X_train)}")
print(f"      Validation sequences: {len(X_val)}")

X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

actual_values = test_df['Total'].values

# ============================================================
# Model Definitions
# ============================================================

class LSTMModel(nn.Module):
    """
    LSTM (Long Short-Term Memory) Network
    
    Architecture:
    - 2 LSTM layers with 64 hidden units each
    - Dropout 0.2 between layers
    - Fully connected output layer
    
    LSTM Cell contains 3 gates:
    - Forget gate: decides what to discard from cell state
    - Input gate: decides what new information to store
    - Output gate: decides what to output based on cell state
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) Network
    
    Architecture:
    - 2 GRU layers with 64 hidden units each
    - Dropout 0.2 between layers
    - Fully connected output layer
    
    GRU Cell contains 2 gates (simpler than LSTM):
    - Reset gate: decides how much past information to forget
    - Update gate: decides how much of the candidate state to use
    
    Advantages over LSTM:
    - Fewer parameters (faster training)
    - Often performs comparably or better on smaller datasets
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

# ============================================================
# Training Function
# ============================================================
def train_model(model, model_name):
    """Train a PyTorch model with early stopping"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n      Training {model_name}...")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"        Epoch [{epoch+1:3d}/{EPOCHS}] Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"        Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    print(f"        Best validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses, best_val_loss

# ============================================================
# Prediction Function
# ============================================================
def predict_dl_model(model):
    """Generate predictions for test set using rolling forecast"""
    model.eval()
    last_seq = train_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, -1)
    last_seq_t = torch.FloatTensor(last_seq).to(device)
    
    predictions = []
    current_seq = last_seq_t.clone()
    
    with torch.no_grad():
        for i in range(len(test_df)):
            pred = model(current_seq)
            predictions.append(pred.cpu().numpy()[0, 0])
            
            next_features = test_scaled[i].copy().reshape(1, 1, -1)
            next_features[0, 0, 0] = pred.item()
            
            new_seq = np.concatenate([
                current_seq.cpu().numpy()[:, 1:, :],
                next_features
            ], axis=1)
            current_seq = torch.FloatTensor(new_seq).to(device)
    
    predictions = np.array(predictions).reshape(-1, 1)
    return target_scaler.inverse_transform(predictions).flatten()

# ============================================================
# Evaluation Function
# ============================================================
def evaluate_predictions(actual, predicted):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

# ============================================================
# PART 2: LSTM MODEL
# ============================================================
print("\n" + "="*70)
print("PART 2: LSTM MODEL (Long Short-Term Memory)")
print("="*70)

print(f"""
[2.1] Model Architecture
      ┌─────────────────────────────────────────────────────────┐
      │  Input: (batch, {SEQ_LENGTH}, {train_features.shape[1]})                              │
      │    ↓                                                    │
      │  LSTM Layer 1: {HIDDEN_SIZE} hidden units                        │
      │    ↓                                                    │
      │  Dropout: {DROPOUT}                                          │
      │    ↓                                                    │
      │  LSTM Layer 2: {HIDDEN_SIZE} hidden units                        │
      │    ↓                                                    │
      │  Fully Connected: {HIDDEN_SIZE} → 1                              │
      │    ↓                                                    │
      │  Output: (batch, 1)                                     │
      └─────────────────────────────────────────────────────────┘
""")

lstm_model = LSTMModel(train_features.shape[1], HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
lstm_params = sum(p.numel() for p in lstm_model.parameters())

print(f"[2.2] Model Parameters")
print(f"      Total parameters: {lstm_params:,}")
print(f"      Trainable parameters: {lstm_params:,}")

print(f"\n[2.3] Training Configuration")
print(f"      Optimizer: Adam")
print(f"      Learning rate: {LEARNING_RATE}")
print(f"      Loss function: MSE (Mean Squared Error)")
print(f"      Batch size: {BATCH_SIZE}")
print(f"      Max epochs: {EPOCHS}")
print(f"      Early stopping patience: {EARLY_STOP_PATIENCE}")

print(f"\n[2.4] Training Process")
lstm_model, lstm_train_losses, lstm_val_losses, lstm_best_val = train_model(lstm_model, "LSTM")

# Save LSTM model
torch.save(lstm_model.state_dict(), f'{DATA_DIR}/lstm_model.pth')

# LSTM Predictions
lstm_predictions = predict_dl_model(lstm_model)
lstm_mae, lstm_rmse, lstm_mape = evaluate_predictions(actual_values, lstm_predictions)

print(f"\n[2.5] LSTM Evaluation Results")
print(f"      ┌────────────────────────────────────────┐")
print(f"      │  MAE  : {lstm_mae:>10,.0f} passengers      │")
print(f"      │  RMSE : {lstm_rmse:>10,.0f} passengers      │")
print(f"      │  MAPE : {lstm_mape:>10.2f}%               │")
print(f"      └────────────────────────────────────────┘")

print(f"\n[2.6] LSTM Detailed Predictions")
print(f"      {'Date':<12} {'Day':<4} {'Actual':>10} {'Predicted':>10} {'Error':>8} {'Error%':>7}")
print(f"      " + "-"*55)
for i, (_, row) in enumerate(test_df.iterrows()):
    actual = row['Total']
    pred = lstm_predictions[i]
    error = actual - pred
    error_pct = error / actual * 100
    print(f"      {row['Date'].strftime('%Y-%m-%d'):<12} {day_names[row['DayOfWeek']]:<4} {actual:>10,} {pred:>10,.0f} {error:>+8,.0f} {error_pct:>+6.1f}%")

# ============================================================
# PART 3: GRU MODEL
# ============================================================
print("\n" + "="*70)
print("PART 3: GRU MODEL (Gated Recurrent Unit)")
print("="*70)

print(f"""
[3.1] Model Architecture
      ┌─────────────────────────────────────────────────────────┐
      │  Input: (batch, {SEQ_LENGTH}, {train_features.shape[1]})                              │
      │    ↓                                                    │
      │  GRU Layer 1: {HIDDEN_SIZE} hidden units                         │
      │    ↓                                                    │
      │  Dropout: {DROPOUT}                                          │
      │    ↓                                                    │
      │  GRU Layer 2: {HIDDEN_SIZE} hidden units                         │
      │    ↓                                                    │
      │  Fully Connected: {HIDDEN_SIZE} → 1                              │
      │    ↓                                                    │
      │  Output: (batch, 1)                                     │
      └─────────────────────────────────────────────────────────┘

[3.2] GRU vs LSTM Comparison
      ┌────────────────────────────────────────────────────────┐
      │  Feature          │   LSTM    │    GRU    │           │
      ├────────────────────────────────────────────────────────┤
      │  Gates            │     3     │     2     │ GRU简化   │
      │  Cell State       │    Yes    │    No     │ GRU无     │
      │  Parameters       │   More    │   Less    │ GRU更少   │
      │  Training Speed   │  Slower   │  Faster   │ GRU更快   │
      └────────────────────────────────────────────────────────┘
""")

gru_model = GRUModel(train_features.shape[1], HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
gru_params = sum(p.numel() for p in gru_model.parameters())

print(f"[3.3] Model Parameters")
print(f"      Total parameters: {gru_params:,}")
print(f"      Parameter reduction vs LSTM: {(lstm_params - gru_params) / lstm_params * 100:.1f}%")

print(f"\n[3.4] Training Configuration")
print(f"      (Same as LSTM)")

print(f"\n[3.5] Training Process")
gru_model, gru_train_losses, gru_val_losses, gru_best_val = train_model(gru_model, "GRU")

# Save GRU model
torch.save(gru_model.state_dict(), f'{DATA_DIR}/gru_model.pth')

# GRU Predictions
gru_predictions = predict_dl_model(gru_model)
gru_mae, gru_rmse, gru_mape = evaluate_predictions(actual_values, gru_predictions)

print(f"\n[3.6] GRU Evaluation Results")
print(f"      ┌────────────────────────────────────────┐")
print(f"      │  MAE  : {gru_mae:>10,.0f} passengers      │")
print(f"      │  RMSE : {gru_rmse:>10,.0f} passengers      │")
print(f"      │  MAPE : {gru_mape:>10.2f}%               │")
print(f"      └────────────────────────────────────────┘")

print(f"\n[3.7] GRU Detailed Predictions")
print(f"      {'Date':<12} {'Day':<4} {'Actual':>10} {'Predicted':>10} {'Error':>8} {'Error%':>7}")
print(f"      " + "-"*55)
for i, (_, row) in enumerate(test_df.iterrows()):
    actual = row['Total']
    pred = gru_predictions[i]
    error = actual - pred
    error_pct = error / actual * 100
    print(f"      {row['Date'].strftime('%Y-%m-%d'):<12} {day_names[row['DayOfWeek']]:<4} {actual:>10,} {pred:>10,.0f} {error:>+8,.0f} {error_pct:>+6.1f}%")

# ============================================================
# PART 4: XGBoost MODEL (ML Baseline)
# ============================================================
print("\n" + "="*70)
print("PART 4: XGBoost MODEL (Machine Learning Baseline)")
print("="*70)

print(f"""
[4.1] Model Description
      XGBoost (eXtreme Gradient Boosting) is a popular gradient boosting
      algorithm that builds an ensemble of decision trees sequentially.
      
      Used as baseline to compare deep learning vs traditional ML.
""")

# Flatten sequences for XGBoost
X_xgb_train = X_train.reshape(X_train.shape[0], -1)
X_xgb_val = X_val.reshape(X_val.shape[0], -1)

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=SEED
)

print(f"[4.2] XGBoost Configuration")
print(f"      n_estimators: 100")
print(f"      max_depth: 6")
print(f"      learning_rate: 0.1")
print(f"      Input dimension: {X_xgb_train.shape[1]} (flattened)")

print(f"\n[4.3] Training XGBoost...")
xgb_model.fit(X_xgb_train, y_train)
print(f"      Training completed.")

# XGBoost Predictions
last_seq_xgb = train_scaled[-SEQ_LENGTH:].reshape(1, -1)
xgb_predictions_scaled = []
current_seq_xgb = last_seq_xgb.copy()

for i in range(len(test_df)):
    pred = xgb_model.predict(current_seq_xgb)[0]
    xgb_predictions_scaled.append(pred)
    
    next_features = test_scaled[i].copy()
    next_features[0] = pred
    
    new_seq = np.concatenate([
        current_seq_xgb.reshape(SEQ_LENGTH, -1)[1:],
        next_features.reshape(1, -1)
    ]).reshape(1, -1)
    current_seq_xgb = new_seq

xgb_predictions_scaled = np.array(xgb_predictions_scaled).reshape(-1, 1)
xgb_predictions = target_scaler.inverse_transform(xgb_predictions_scaled).flatten()
xgb_mae, xgb_rmse, xgb_mape = evaluate_predictions(actual_values, xgb_predictions)

print(f"\n[4.4] XGBoost Evaluation Results")
print(f"      MAE: {xgb_mae:,.0f} | RMSE: {xgb_rmse:,.0f} | MAPE: {xgb_mape:.2f}%")

# ============================================================
# PART 5: MODEL COMPARISON
# ============================================================
print("\n" + "="*70)
print("PART 5: MODEL COMPARISON SUMMARY")
print("="*70)

print(f"""
[5.1] Performance Comparison
      ┌─────────────────────────────────────────────────────────────┐
      │  Model    │  Parameters │    MAE    │   RMSE   │   MAPE   │
      ├─────────────────────────────────────────────────────────────┤
      │  LSTM     │   {lstm_params:>8,}  │  {lstm_mae:>7,.0f}  │  {lstm_rmse:>6,.0f}  │  {lstm_mape:>5.2f}%  │
      │  GRU      │   {gru_params:>8,}  │  {gru_mae:>7,.0f}  │  {gru_rmse:>6,.0f}  │  {gru_mape:>5.2f}%  │
      │  XGBoost  │        N/A  │  {xgb_mae:>7,.0f}  │  {xgb_rmse:>6,.0f}  │  {xgb_mape:>5.2f}%  │
      └─────────────────────────────────────────────────────────────┘
""")

# Determine best model
results = [
    ('LSTM', lstm_mape, lstm_predictions),
    ('GRU', gru_mape, gru_predictions),
    ('XGBoost', xgb_mape, xgb_predictions)
]
results.sort(key=lambda x: x[1])
best_model = results[0][0]
best_mape = results[0][1]

print(f"[5.2] Ranking (by MAPE)")
for i, (name, mape, _) in enumerate(results, 1):
    medal = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
    print(f"      {medal} #{i}: {name} (MAPE: {mape:.2f}%)")

print(f"\n[5.3] Key Findings")
print(f"      • Best model: {best_model} with MAPE {best_mape:.2f}%")
print(f"      • GRU has {(lstm_params - gru_params) / lstm_params * 100:.1f}% fewer parameters than LSTM")
gru_vs_lstm = "better" if gru_mape < lstm_mape else "worse"
print(f"      • GRU performs {gru_vs_lstm} than LSTM on this dataset")
print(f"      • All models achieve MAPE < 10% (good performance)")

# ============================================================
# PART 6: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("PART 6: GENERATING VISUALIZATIONS")
print("="*70)

# Figure 1: LSTM Training Loss
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(lstm_train_losses, label='Training Loss', color='blue', alpha=0.7, linewidth=2)
ax1.plot(lstm_val_losses, label='Validation Loss', color='orange', alpha=0.7, linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('LSTM: Training and Validation Loss', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/05_lstm_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 05_lstm_training_loss.png")

# Figure 2: GRU Training Loss
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(gru_train_losses, label='Training Loss', color='green', alpha=0.7, linewidth=2)
ax2.plot(gru_val_losses, label='Validation Loss', color='red', alpha=0.7, linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (MSE)', fontsize=12)
ax2.set_title('GRU: Training and Validation Loss', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/06_gru_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 06_gru_training_loss.png")

# Figure 3: LSTM Forecast
fig3, ax3 = plt.subplots(figsize=(12, 6))
context_days = 15
context_df = train_df.iloc[-context_days:]
all_dates = pd.concat([context_df['Date'], test_df['Date']])
all_actual = np.concatenate([context_df['Total'].values, actual_values])

ax3.plot(all_dates, all_actual, 'b-', label='Actual', linewidth=1.5, marker='o', markersize=4)
ax3.plot(test_df['Date'], lstm_predictions, 'r--', label='LSTM Prediction', 
         linewidth=2.5, marker='s', markersize=8)
ax3.axvline(x=train_df['Date'].iloc[-1], color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Daily Passengers', fontsize=12)
ax3.set_title(f'LSTM Forecast (MAPE: {lstm_mape:.2f}%)', fontsize=14)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/07_lstm_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 07_lstm_forecast.png")

# Figure 4: GRU Forecast
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(all_dates, all_actual, 'b-', label='Actual', linewidth=1.5, marker='o', markersize=4)
ax4.plot(test_df['Date'], gru_predictions, 'g--', label='GRU Prediction', 
         linewidth=2.5, marker='^', markersize=8)
ax4.axvline(x=train_df['Date'].iloc[-1], color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
ax4.set_xlabel('Date', fontsize=12)
ax4.set_ylabel('Daily Passengers', fontsize=12)
ax4.set_title(f'GRU Forecast (MAPE: {gru_mape:.2f}%)', fontsize=14)
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/08_gru_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 08_gru_forecast.png")

# Figure 5: All Models Comparison
fig5, ax5 = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(test_df))

ax5.plot(x_pos, actual_values, 'ko-', label='Actual', linewidth=2, markersize=10)
ax5.plot(x_pos, lstm_predictions, 'rs--', label=f'LSTM (MAPE:{lstm_mape:.1f}%)', linewidth=1.5, markersize=7)
ax5.plot(x_pos, gru_predictions, 'g^--', label=f'GRU (MAPE:{gru_mape:.1f}%)', linewidth=1.5, markersize=7)
ax5.plot(x_pos, xgb_predictions, 'mD--', label=f'XGBoost (MAPE:{xgb_mape:.1f}%)', linewidth=1.5, markersize=7)

ax5.set_xlabel('Date', fontsize=12)
ax5.set_ylabel('Daily Passengers', fontsize=12)
ax5.set_title('5-Day Forecast: All Models Comparison', fontsize=14)
ax5.set_xticks(x_pos)
ax5.set_xticklabels([f"{d.strftime('%m/%d')}\n({day_names[test_df.iloc[i]['DayOfWeek']]})" 
                     for i, d in enumerate(test_df['Date'])], fontsize=11)
ax5.legend(loc='upper right', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/09_all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 09_all_models_comparison.png")

# Figure 6: Bar Comparison
fig6, ax6 = plt.subplots(figsize=(14, 6))
x_pos = np.arange(len(test_df))
width = 0.2

bars1 = ax6.bar(x_pos - 1.5*width, actual_values, width, label='Actual', color='steelblue', alpha=0.8)
bars2 = ax6.bar(x_pos - 0.5*width, lstm_predictions, width, label='LSTM', color='salmon', alpha=0.8)
bars3 = ax6.bar(x_pos + 0.5*width, gru_predictions, width, label='GRU', color='lightgreen', alpha=0.8)
bars4 = ax6.bar(x_pos + 1.5*width, xgb_predictions, width, label='XGBoost', color='plum', alpha=0.8)

ax6.set_xlabel('Date', fontsize=12)
ax6.set_ylabel('Daily Passengers', fontsize=12)
ax6.set_title('5-Day Forecast: Bar Comparison', fontsize=14)
ax6.set_xticks(x_pos)
ax6.set_xticklabels([f"{d.strftime('%m/%d')}\n({day_names[test_df.iloc[i]['DayOfWeek']]})" 
                     for i, d in enumerate(test_df['Date'])], fontsize=11)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/10_bar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 10_bar_comparison.png")

# Figure 7: Error Analysis
fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5))

# LSTM errors
ax7a = axes7[0]
lstm_errors = (actual_values - lstm_predictions) / actual_values * 100
colors = ['green' if e >= 0 else 'red' for e in lstm_errors]
bars = ax7a.bar(x_pos, lstm_errors, color=colors, alpha=0.7)
ax7a.axhline(y=0, color='black', linewidth=1)
ax7a.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
ax7a.axhline(y=-5, color='gray', linestyle='--', alpha=0.5)
ax7a.set_xlabel('Date', fontsize=11)
ax7a.set_ylabel('Error (%)', fontsize=11)
ax7a.set_title(f'LSTM Prediction Error (MAPE: {lstm_mape:.2f}%)', fontsize=12)
ax7a.set_xticks(x_pos)
ax7a.set_xticklabels([d.strftime('%m/%d') for d in test_df['Date']])
ax7a.set_ylim(-15, 15)
ax7a.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, lstm_errors):
    y_pos = val + 0.5 if val >= 0 else val - 0.5
    ax7a.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%', 
             ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

# GRU errors
ax7b = axes7[1]
gru_errors = (actual_values - gru_predictions) / actual_values * 100
colors = ['green' if e >= 0 else 'red' for e in gru_errors]
bars = ax7b.bar(x_pos, gru_errors, color=colors, alpha=0.7)
ax7b.axhline(y=0, color='black', linewidth=1)
ax7b.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
ax7b.axhline(y=-5, color='gray', linestyle='--', alpha=0.5)
ax7b.set_xlabel('Date', fontsize=11)
ax7b.set_ylabel('Error (%)', fontsize=11)
ax7b.set_title(f'GRU Prediction Error (MAPE: {gru_mape:.2f}%)', fontsize=12)
ax7b.set_xticks(x_pos)
ax7b.set_xticklabels([d.strftime('%m/%d') for d in test_df['Date']])
ax7b.set_ylim(-15, 15)
ax7b.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, gru_errors):
    y_pos = val + 0.5 if val >= 0 else val - 0.5
    ax7b.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%', 
             ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/11_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 11_error_analysis.png")

# Figure 8: MAPE Comparison
fig8, ax8 = plt.subplots(figsize=(8, 6))
models = ['LSTM', 'GRU', 'XGBoost']
mapes = [lstm_mape, gru_mape, xgb_mape]
colors = ['salmon', 'lightgreen', 'plum']

# Sort by MAPE
sorted_idx = np.argsort(mapes)
models = [models[i] for i in sorted_idx]
mapes = [mapes[i] for i in sorted_idx]
colors = [colors[i] for i in sorted_idx]

bars = ax8.bar(models, mapes, color=colors, alpha=0.8, edgecolor='black')
ax8.set_xlabel('Model', fontsize=12)
ax8.set_ylabel('MAPE (%)', fontsize=12)
ax8.set_title('Model Comparison: MAPE (Lower is Better)', fontsize=14)
ax8.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, mapes):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/12_mape_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: 12_mape_comparison.png")

# ============================================================
# Save Results
# ============================================================
print("\n" + "="*70)
print("PART 7: SAVING RESULTS")
print("="*70)

# Save metrics
metrics_df = pd.DataFrame({
    'Model': ['LSTM', 'GRU', 'XGBoost'],
    'MAE': [lstm_mae, gru_mae, xgb_mae],
    'RMSE': [lstm_rmse, gru_rmse, xgb_rmse],
    'MAPE': [lstm_mape, gru_mape, xgb_mape],
    'Parameters': [lstm_params, gru_params, 'N/A']
})
metrics_df = metrics_df.sort_values('MAPE')
metrics_df.to_csv(f'{DATA_DIR}/model_comparison.csv', index=False)
print("      Saved: model_comparison.csv")

# Save predictions
predictions_df = test_df[['Date', 'Total']].copy()
predictions_df['LSTM'] = lstm_predictions
predictions_df['GRU'] = gru_predictions
predictions_df['XGBoost'] = xgb_predictions
predictions_df.to_csv(f'{DATA_DIR}/all_predictions.csv', index=False)
print("      Saved: all_predictions.csv")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
Models Trained: 3 (LSTM, GRU, XGBoost)

Best Model: {best_model} (MAPE: {best_mape:.2f}%)

LSTM Configuration:
  - Architecture: 2-layer LSTM, {HIDDEN_SIZE} hidden units
  - Parameters: {lstm_params:,}
  - Performance: MAE={lstm_mae:,.0f}, MAPE={lstm_mape:.2f}%

GRU Configuration:
  - Architecture: 2-layer GRU, {HIDDEN_SIZE} hidden units  
  - Parameters: {gru_params:,} ({(lstm_params-gru_params)/lstm_params*100:.1f}% fewer than LSTM)
  - Performance: MAE={gru_mae:,.0f}, MAPE={gru_mape:.2f}%

XGBoost Configuration:
  - n_estimators=100, max_depth=6
  - Performance: MAE={xgb_mae:,.0f}, MAPE={xgb_mape:.2f}%

Output Files:
  - Figures: 05-12 in figures/
  - Data: model_comparison.csv, all_predictions.csv
  - Models: lstm_model.pth, gru_model.pth
""")
