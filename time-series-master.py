"""
11_time_series_pipeline.py
Comprehensive Time Series Analysis and Forecasting Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPreprocessor:
    """Comprehensive time series preprocessing"""
    
    def __init__(self):
        self.scaler = None
        self.feature_scalers = {}
        
    def create_time_features(self, df, date_column):
        """Create comprehensive time-based features"""
        df_temp = df.copy()
        df_temp[date_column] = pd.to_datetime(df_temp[date_column])
        
        # Basic time features
        df_temp['year'] = df_temp[date_column].dt.year
        df_temp['month'] = df_temp[date_column].dt.month
        df_temp['week'] = df_temp[date_column].dt.isocalendar().week
        df_temp['day'] = df_temp[date_column].dt.day
        df_temp['dayofweek'] = df_temp[date_column].dt.dayofweek
        df_temp['dayofyear'] = df_temp[date_column].dt.dayofyear
        df_temp['quarter'] = df_temp[date_column].dt.quarter
        df_temp['is_weekend'] = (df_temp[date_column].dt.dayofweek >= 5).astype(int)
        df_temp['is_month_start'] = df_temp[date_column].dt.is_month_start.astype(int)
        df_temp['is_month_end'] = df_temp[date_column].dt.is_month_end.astype(int)
        
        # Cyclical features
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        df_temp['day_sin'] = np.sin(2 * np.pi * df_temp['day'] / 31)
        df_temp['day_cos'] = np.cos(2 * np.pi * df_temp['day'] / 31)
        df_temp['dayofweek_sin'] = np.sin(2 * np.pi * df_temp['dayofweek'] / 7)
        df_temp['dayofweek_cos'] = np.cos(2 * np.pi * df_temp['dayofweek'] / 7)
        
        return df_temp
    
    def create_lag_features(self, df, value_column, lags=[1, 7, 30, 90]):
        """Create lag features"""
        df_lags = df.copy()
        
        for lag in lags:
            df_lags[f'{value_column}_lag_{lag}'] = df[value_column].shift(lag)
        
        return df_lags
    
    def create_rolling_features(self, df, value_column, windows=[7, 30, 90]):
        """Create rolling statistics"""
        df_rolling = df.copy()
        
        for window in windows:
            df_rolling[f'{value_column}_roll_mean_{window}'] = df[value_column].rolling(window=window, min_periods=1).mean()
            df_rolling[f'{value_column}_roll_std_{window}'] = df[value_column].rolling(window=window, min_periods=1).std()
            df_rolling[f'{value_column}_roll_min_{window}'] = df[value_column].rolling(window=window, min_periods=1).min()
            df_rolling[f'{value_column}_roll_max_{window}'] = df[value_column].rolling(window=window, min_periods=1).max()
            df_rolling[f'{value_column}_roll_median_{window}'] = df[value_column].rolling(window=window, min_periods=1).median()
        
        return df_rolling
    
    def create_expanding_features(self, df, value_column):
        """Create expanding window features"""
        df_expanding = df.copy()
        
        df_expanding[f'{value_column}_expanding_mean'] = df[value_column].expanding().mean()
        df_expanding[f'{value_column}_expanding_std'] = df[value_column].expanding().std()
        df_expanding[f'{value_column}_expanding_min'] = df[value_column].expanding().min()
        df_expanding[f'{value_column}_expanding_max'] = df[value_column].expanding().max()
        
        return df_expanding
    
    def check_stationarity(self, series, alpha=0.05):
        """Check stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < alpha
        
        return {
            'p_value': p_value,
            'is_stationary': is_stationary,
            'test_statistic': result[0],
            'critical_values': result[4]
        }
    
    def make_stationary(self, series, method='diff'):
        """Make time series stationary"""
        if method == 'diff':
            return series.diff().dropna()
        elif method == 'log_diff':
            return np.log(series).diff().dropna()
        elif method == 'seasonal_diff':
            return series.diff(periods=12).dropna()  # Assuming monthly data
        else:
            raise ValueError("Method must be 'diff', 'log_diff', or 'seasonal_diff'")
    
    def decompose_time_series(self, series, period=12, model='additive'):
        """Decompose time series into trend, seasonal, and residual components"""
        decomposition = seasonal_decompose(series, period=period, model=model)
        return decomposition
    
    def scale_data(self, data, method='standard'):
        """Scale time series data"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        return scaled_data
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for time series forecasting"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            targets.append(data[i + sequence_length])
        
        return np.array(sequences), np.array(targets)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """LSTM for time series forecasting"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Output layer
        output = self.linear(last_hidden)
        return output

class GRUModel(nn.Module):
    """GRU for time series forecasting"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # GRU forward
        gru_out, hn = self.gru(x, h0)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Output layer
        output = self.linear(last_hidden)
        return output

class TransformerTimeSeries(nn.Module):
    """Transformer for time series forecasting"""
    
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=3, output_size=1, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncodingTS(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Use last time step
        x = x[:, -1, :]
        output = self.output_layer(x)
        return output

class PositionalEncodingTS(nn.Module):
    """Positional encoding for time series transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingTS, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CNN1DTimeSeries(nn.Module):
    """1D CNN for time series forecasting"""
    
    def __init__(self, input_size=1, output_size=1, sequence_length=60):
        super(CNN1DTimeSeries, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Calculate flattened size
        self.flatten_size = 256
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TimeSeriesTrainer:
    """Time series model trainer"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
        """Train time series model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self.evaluate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            train_loss_avg = train_loss / len(train_loader)
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss_avg:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def evaluate(self, data_loader, criterion):
        """Evaluate model"""
        self.model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
        
        return test_loss / len(data_loader)
    
    def predict(self, data_loader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.numpy())
        
        return np.array(predictions), np.array(actuals)

class TimeSeriesPipeline:
    """Complete time series forecasting pipeline"""
    
    def __init__(self, sequence_length=60, model_type='lstm'):
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.preprocessor = TimeSeriesPreprocessor()
        self.model = None
        self.scaler = None
        
    def prepare_data(self, series, date_index=None, test_size=0.2):
        """Prepare time series data for training"""
        # Convert to DataFrame if needed
        if isinstance(series, pd.Series):
            df = series.to_frame()
        else:
            df = series.copy()
        
        # Add time features if date index provided
        if date_index is not None:
            df = self.preprocessor.create_time_features(df, date_index)
        
        # Scale the target variable
        target_column = series.name if isinstance(series, pd.Series) else df.columns[0]
        scaled_series = self.preprocessor.scale_data(df[target_column].values, method='minmax')
        self.scaler = self.preprocessor.scaler
        
        # Create sequences
        sequences, targets = self.preprocessor.create_sequences(scaled_series, self.sequence_length)
        
        # Split data
        split_idx = int(len(sequences) * (1 - test_size))
        X_train, X_test = sequences[:split_idx], sequences[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        return train_dataset, test_dataset, scaled_series
    
    def create_model(self, input_size=1, output_size=1, **kwargs):
        """Create time series model"""
        if self.model_type == 'lstm':
            self.model = LSTMModel(input_size=input_size, output_size=output_size, **kwargs)
        elif self.model_type == 'gru':
            self.model = GRUModel(input_size=input_size, output_size=output_size, **kwargs)
        elif self.model_type == 'transformer':
            self.model = TransformerTimeSeries(input_size=input_size, output_size=output_size, **kwargs)
        elif self.model_type == 'cnn':
            self.model = CNN1DTimeSeries(input_size=input_size, output_size=output_size, **kwargs)
        else:
            raise ValueError("Model type must be 'lstm', 'gru', 'transformer', or 'cnn'")
        
        return self.model
    
    def train(self, train_dataset, val_dataset, epochs=100, batch_size=32, **kwargs):
        """Train the time series model"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = TimeSeriesTrainer(self.model, device)
        
        history = trainer.train(train_loader, val_loader, epochs=epochs, **kwargs)
        return history
    
    def forecast(self, test_dataset, batch_size=32):
        """Make forecasts"""
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = TimeSeriesTrainer(self.model, device)
        
        predictions, actuals = trainer.predict(test_loader)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = self.scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        return predictions, actuals
    
    def evaluate_forecast(self, predictions, actuals):
        """Evaluate forecast performance"""
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def plot_forecast(self, predictions, actuals, title="Time Series Forecast"):
        """Plot forecast vs actual"""
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample time series data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    trend = np.linspace(0, 10, 1000)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(1000) / 365)
    noise = np.random.normal(0, 0.5, 1000)
    values = trend + seasonal + noise
    
    series = pd.Series(values, index=dates, name='value')
    
    # Initialize pipeline
    pipeline = TimeSeriesPipeline(sequence_length=60, model_type='lstm')
    
    # Prepare data
    train_dataset, test_dataset, scaled_series = pipeline.prepare_data(series, date_index=series.index.name)
    
    # Create and train model
    pipeline.create_model(input_size=1, output_size=1, hidden_size=50)
    history = pipeline.train(train_dataset, test_dataset, epochs=50, batch_size=32)
    
    # Make forecasts
    predictions, actuals = pipeline.forecast(test_dataset)
    
    # Evaluate
    metrics = pipeline.evaluate_forecast(predictions, actuals)
    print("Forecast Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    pipeline.plot_forecast(predictions, actuals)
    
    print("Time Series Pipeline Ready!")