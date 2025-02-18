import pandas as pd
import numpy as np
import time
import requests
import pygame
import sys
import os
import joblib
from threading import Thread
from datetime import datetime
from pytz import timezone
import talib as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats

# Audio Configuration
THEME_SOUND = 'themebot.mp3'
CALL_SOUND = 'CALLS.mp3'
PUT_SOUND = 'PUTS.mp3'

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(2)

class MachineLearningModel:
    """Enhanced ML model with training/live data comparison"""
    
    def __init__(self):
        self.models = {}
        self.training_stats = {}  # Stores feature statistics
        self.features = ['RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower', 
                        'EMA_8', 'VWAP', 'OBV', 'Fib_38_2', 'Fib_61_8']
        self.csv_files = ["AAPL1.csv", "NVDA1.csv", "TSLA1.csv"]
        
        for csv_file in self.csv_files:
            model_filename = f"model_{os.path.basename(csv_file).replace('.csv', '.pkl')}"
            if os.path.exists(model_filename):
                print(f"Loading existing model for {csv_file}...")
                try:
                    loaded = joblib.load(model_filename)
                    
                    # Check if loaded model has proper format
                    if isinstance(loaded, dict) and 'model' in loaded and 'stats' in loaded:
                        self.models[csv_file] = loaded['model']
                        self.training_stats[csv_file] = loaded['stats']
                        print(f"Successfully loaded new format model for {csv_file}")
                    else:
                        # Handle old model format by retraining
                        print(f"⚠️ Old model format detected for {csv_file}, retraining...")
                        df = self.load_and_preprocess_data(csv_file)
                        self.train_model(csv_file, df)
                except Exception as e:
                    print(f"Error loading model {csv_file}: {e}, retraining...")
                    df = self.load_and_preprocess_data(csv_file)
                    self.train_model(csv_file, df)
            else:
                print(f"Processing {csv_file}...")
                df = self.load_and_preprocess_data(csv_file)
                self.train_model(csv_file, df)

    def load_and_preprocess_data(self, csv_file):
        """Load and preprocess CSV data for model training"""
        try:
            df = pd.read_csv(csv_file)
            
            # Clean column names (assuming Alpha Vantage format)
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            })
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Calculate technical indicators
            df = calculate_indicators(df)
            
            # Calculate Fibonacci levels
            df = calculate_fibonacci_levels(df)
            
            return df.dropna()
        except Exception as e:
            print(f"Error loading/preprocessing {csv_file}: {e}")
            raise

    def train_model(self, file_path, df):
        """Enhanced training with statistical preservation"""
        X = df[self.features]
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # Changed '4. close' to 'close'

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # Calculate and store training statistics
        train_stats = {
            'means': X_train.mean().to_dict(),
            'stds': X_train.std().to_dict(),
            'kde': {feature: stats.gaussian_kde(X_train[feature]) for feature in self.features}
        }
    
        model_filename = f"model_{os.path.basename(file_path).replace('.csv', '.pkl')}"
        joblib.dump({'model': model, 'stats': train_stats}, model_filename)
        self.models[file_path] = model
        self.training_stats[file_path] = train_stats
        print(f"Model and stats saved for {file_path}")

    def compare_live_data(self, file_path, live_features):
        """Compare live data against training distribution"""
        stats = self.training_stats.get(file_path)
        if not stats:
            print("No training statistics available")
            return None

        report = {
            'z_scores': {},
            'probability_density': {},
            'anomalies': []
        }

        for feature in self.features:
            # Z-score analysis
            live_value = live_features.get(feature, 0)
            mean = stats['means'][feature]
            std = stats['stds'][feature]
            
            if std > 0:
                z = (live_value - mean) / std
                report['z_scores'][feature] = z
                if abs(z) > 3:
                    report['anomalies'].append(feature)
            
            # Probability density analysis
            kde = stats['kde'][feature]
            report['probability_density'][feature] = kde.integrate_box_1d(live_value-0.01, live_value+0.01)

        return report

    def predict_trend(self, file_path, latest_data):
        """Enhanced prediction with data validation"""
        if file_path not in self.models:
            print(f"No model found for {file_path}")
            return None

        # Create feature dictionary
        live_features = dict(zip(self.features, latest_data))
        
        # Perform data comparison
        comparison = self.compare_live_data(file_path, live_features)
        if comparison:
            print(f"\nData Validation Report for {file_path}:")
            print(f"Significant anomalies: {comparison['anomalies'] or 'None'}")
            for feat in self.features:
                print(f"{feat}: Z={comparison['z_scores'][feat]:.2f} | PD={comparison['probability_density'][feat]:.4f}")

        # Make prediction
        model = self.models[file_path]
        latest_df = pd.DataFrame([latest_data], columns=self.features)
        return model.predict(latest_df)

def play_theme_loop():
    """Play theme music continuously in a separate thread"""
    try:
        pygame.mixer.music.load(THEME_SOUND)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Theme music error: {e}")

def play_signal_sound(symbol, signal_type):
    """Play trade signal sounds without interrupting theme"""
    try:
        sound = pygame.mixer.Sound(CALL_SOUND if signal_type == 'call' else PUT_SOUND)
        pygame.mixer.Channel(1).play(sound)
        print(f"Signal Triggered: {symbol} - {signal_type.upper()} Trade")
    except Exception as e:
        print(f"Signal sound error: {e}")

# Alpha Vantage API Configuration
API_KEY = 'F2B18RC451038ETB'
SYMBOLS = ['NVDA', 'AAPL', 'TSLA']
INTERVAL = '5min'
URL_TEMPLATE = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&entitlement=realtime&apikey={api_key}"

def fetch_realtime_data(symbol):
    """Fetch real-time 5-minute intraday data with correct timezone handling"""
    try:
        response = requests.get(URL_TEMPLATE.format(
            symbol=symbol,
            interval=INTERVAL,
            api_key=API_KEY
        ))
        data = response.json()

        if "Time Series (5min)" not in data:
            return None

        df = pd.DataFrame.from_dict(data["Time Series (5min)"], orient="index")
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        df.index = pd.to_datetime(df.index).tz_localize('America/New_York')
        df = df.astype(float).sort_index()
        return df
    except Exception as e:
        print(f"Data fetch error: {e}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators with validation"""
    try:
        closes = df['close'].astype(float)
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        volumes = df['volume'].astype(float)

        df['RSI'] = ta.RSI(closes)
        macd, macd_signal, _ = ta.MACD(closes)
        df['MACD'] = macd
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = ta.BBANDS(closes)
        df['EMA_8'] = ta.EMA(closes, timeperiod=8)
        df['VWAP'] = (volumes * (highs + lows + closes) / 3).cumsum() / volumes.cumsum()
        df['OBV'] = ta.OBV(closes, volumes)
        
        return df.dropna()
    except Exception as e:
        print(f"Indicator error: {e}")
        return df

def calculate_fibonacci_levels(df, period=14):
    """Calculate Fibonacci retracement levels"""
    try:
        high = df['high'].rolling(period).max()
        low = df['low'].rolling(period).min()
        diff = high - low
        df['Fib_38_2'] = high - diff * 0.382
        df['Fib_61_8'] = high - diff * 0.618
        return df
    except Exception as e:
        print(f"Fibonacci error: {e}")
        return df

class RealtimeStrategyEngine:
    """Strategy engine with error-protected methods"""
    
    def generate_signals_strategy1(self, df):
        if all(col in df.columns for col in ['RSI', 'MACD', 'MACD_Signal', 'Fib_38_2', 'Fib_61_8', 'close']):
            df['Signal_RSI'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] > 65, -1, 0))
            df['Signal_MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
            df['Signal_Fib'] = np.where((df['close'] < df['Fib_38_2']) & (df['close'] > df['Fib_61_8']), 1, 0)
            df['Buy_Strategy1'] = (df['Signal_RSI'] == 1) & (df['Signal_MACD'] == 1) & (df['Signal_Fib'] == 1)
            df['Sell_Strategy1'] = (df['Signal_RSI'] == -1) & (df['Signal_MACD'] == -1) & (df['Signal_Fib'] == 1)
        return df
    
    def generate_signals_strategy2(self, df):
        if all(col in df.columns for col in ['RSI', 'MACD', 'MACD_Signal']):
            df['Signal_RSI'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] > 65, -1, 0))
            df['Signal_MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
            df['Buy_Strategy2'] = (df['Signal_RSI'] == 1) & (df['Signal_MACD'] == 1)
            df['Sell_Strategy2'] = (df['Signal_RSI'] == -1) & (df['Signal_MACD'] == -1)
        return df
    
    def generate_signals_strategy3(self, df):
        if all(col in df.columns for col in ['VWAP', 'OBV', 'Fib_38_2', 'Fib_61_8', 'close']):
            df['Signal_VWAP'] = np.where(df['close'] > df['VWAP'], 1, -1)
            df['Signal_OBV'] = np.where(df['OBV'].diff() > 0, 1, -1)
            df['Signal_Fib'] = np.where((df['close'] < df['Fib_38_2']) & (df['close'] > df['Fib_61_8']), 1, 0)
            df['Buy_Strategy3'] = (df['Signal_VWAP'] == 1) & (df['Signal_OBV'] == 1) & (df['Signal_Fib'] == 1)
            df['Sell_Strategy3'] = (df['Signal_VWAP'] == -1) & (df['Signal_OBV'] == -1) & (df['Signal_Fib'] == 1)
        return df

    def generate_signals_strategy4(self, df):
        if all(col in df.columns for col in ['RSI', 'MACD', 'MACD_Signal', 'OBV']):
            df['Signal_RSI'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] > 65, -1, 0))
            df['Signal_MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
            df['Signal_OBV'] = np.where(df['OBV'] < df['RSI'], 1, 0)

            df['Buy_Strategy4'] = (df['Signal_RSI'] == 1) & (df['Signal_MACD'] == 1) & (df['Signal_OBV'] == 1)
            df['Sell_Strategy4'] = (df['Signal_RSI'] == -1) & (df['Signal_MACD'] == -1) & (df['Signal_OBV'] == 1)
        return df

    def generate_signals_strategy5(self, df):
        """Strategy 5: EMA Cross + Fibonacci + ATR + ADX (No VWAP, OBV, or Bollinger)"""
        if all(col in df.columns for col in ['EMA_8', 'EMA_21', 'Fib_38_2', 'Fib_61_8', 'ATR', 'ADX']):
            df['Signal_EMA_Cross'] = np.where(df['EMA_8'] > df['EMA_21'], 1, -1)
            df['Signal_ATR'] = np.where(df['ATR'] > df['ATR'].rolling(10).mean(), 1, 0)  # ATR rising
            df['Signal_ADX'] = np.where(df['ADX'] > 20, 1, 0)  # ADX > 20 confirms strong trend

            df['Buy_Strategy5'] = (df['Signal_EMA_Cross'] == 1) & (df['Signal_ATR'] == 1) & (df['Signal_ADX'] == 1)
            df['Sell_Strategy5'] = (df['Signal_EMA_Cross'] == -1) & (df['Signal_ATR'] == 1) & (df['Signal_ADX'] == 1)
        return df

        
# In the main() function, modify the ML prediction section as follows:

def main():
    """Main loop with enhanced ML validation"""
    Thread(target=play_theme_loop, daemon=True).start()
    engine = RealtimeStrategyEngine()
    ml_model = MachineLearningModel()
    
    while True:
        for symbol in SYMBOLS:
            df = fetch_realtime_data(symbol)
            if df is not None and len(df) > 20:
                df = calculate_indicators(df)
                df = calculate_fibonacci_levels(df)
                df = df.sort_index()
                
                # Existing strategy signals
                df = engine.generate_signals_strategy1(df)
                df = engine.generate_signals_strategy2(df)
                df = engine.generate_signals_strategy3(df)
                df = engine.generate_signals_strategy4(df)
                df = engine.generate_signals_strategy5(df)
                
                # ML Prediction for all symbols
                latest_row = df.iloc[-1]
                latest_data = [latest_row[feature] for feature in ml_model.features]
                csv_file = f"{symbol}1.csv"
                
                prediction = ml_model.predict_trend(csv_file, latest_data)
                
                if prediction is not None:
                    trend = 'UPWARD' if prediction == 1 else 'DOWNWARD'
                    print(f"\n{symbol} - ML Prediction: {trend} Trend (Validated)")
                    # Removed play_signal_sound call to prevent ML-based signals
                
                # Process strategy signals
                timestamp = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n=== {symbol} Market Update @ {timestamp} ===")
                
                for strategy in [1, 2, 3, 4]:
                    if df.iloc[-1].get(f'Buy_Strategy{strategy}', False):
                        print(f"STRATEGY {strategy}: BUY CALLS")
                        play_signal_sound(symbol, 'call')
                    if df.iloc[-1].get(f'Sell_Strategy{strategy}', False):
                        print(f"STRATEGY {strategy}: BUY PUTS")
                        play_signal_sound(symbol, 'put')
        
        time.sleep(5)

if __name__ == "__main__":
    main()