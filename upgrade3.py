import pandas as pd
import numpy as np
import time
import requests
import pygame
import sys
from threading import Thread
from datetime import datetime
from pytz import timezone
import talib as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Audio Configuration
THEME_SOUND = 'themebot.mp3'
CALL_SOUND = 'CALLS.mp3'
PUT_SOUND = 'PUTS.mp3'

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(2)

class MachineLearningModel:
    """Machine learning model for trend prediction on AAPL1, NVDA1, and TSLA1 CSV files."""
    
    def __init__(self):
        """Initialize the model for specific CSV files."""
        self.models = {}  # Store separate models per CSV file
        self.features = ['RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower', 'EMA_8', 'VWAP', 'OBV', 'Fib_38_2', 'Fib_61_8']
        self.csv_files = ["AAPL1.csv", "NVDA1.csv", "TSLA1.csv"]
        
        for csv_file in self.csv_files:
            model_filename = f"model_{os.path.basename(csv_file).replace('.csv', '.pkl')}"
            if os.path.exists(model_filename):
                print(f"Loading existing model for {csv_file}...")
                self.models[csv_file] = joblib.load(model_filename)
            else:
                print(f"Processing {csv_file}...")
                df = self.load_and_preprocess_data(csv_file)
                self.train_model(csv_file, df)
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess data from CSV file."""
        df = pd.read_csv(file_path, parse_dates=["Unnamed: 0"])
        df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
        df.set_index("timestamp", inplace=True)
        
        # Compute indicators
        df['RSI'] = ta.RSI(df['4. close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['4. close'])
        df['Bollinger_Upper'], _, df['Bollinger_Lower'] = ta.BBANDS(df['4. close'])
        df['EMA_8'] = ta.EMA(df['4. close'], timeperiod=8)
        df['VWAP'] = (df['4. close'] * df['5. volume']).cumsum() / df['5. volume'].cumsum()
        df['OBV'] = ta.OBV(df['4. close'], df['5. volume'])
        
        # Fibonacci levels (computed from high/low range)
        high = df['2. high'].rolling(window=14).max()
        low = df['3. low'].rolling(window=14).min()
        df['Fib_38_2'] = low + (high - low) * 0.382
        df['Fib_61_8'] = low + (high - low) * 0.618
        
        df.dropna(inplace=True)
        return df
    
    def train_model(self, file_path, df):
        """Train an ML model and save it separately for each CSV file."""
        print(f"Training model for {file_path}...")
        
        # Define features and target labels
        X = df[self.features]
        y = np.where(df['4. close'].shift(-1) > df['4. close'], 1, 0)  # Binary classification: Up (1) or Down (0)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save the trained model with a unique name per CSV file
        model_filename = f"model_{os.path.basename(file_path).replace('.csv', '.pkl')}"
        joblib.dump(model, model_filename)
        self.models[file_path] = model
        print(f"Model trained and saved as {model_filename}")

    def predict_trend(self, file_path, latest_data):
        """Predict the trend using the trained model for the specified file."""
        if file_path not in self.models:
            print(f"No model found for {file_path}. Train the model first.")
            return None
        
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
        url = URL_TEMPLATE.format(
            symbol=symbol,
            interval=INTERVAL,
            api_key=API_KEY
        )
        response = requests.get(url)
        data = response.json()

        if "Time Series (5min)" not in data:
            print(f"Error fetching data for {symbol}:", data)
            return None

        df = pd.DataFrame.from_dict(data["Time Series (5min)"], orient="index")
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        # Correct timezone handling
        df.index = pd.to_datetime(df.index).tz_localize('America/New_York')
        df = df.astype(float)
        df.sort_index(inplace=True)
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators with validation"""
    try:
        closes = df['close'].astype(float)
        if closes.isnull().any():
            print("Missing close prices")
            return None
            
        highs = df['high'].astype(float)
        lows = df['low'].astype(float)
        volumes = df['volume'].astype(float)

        df['RSI'] = ta.RSI(closes)
        macd, macd_signal, _ = ta.MACD(closes)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['Bollinger_Upper'], df['Bollinger_Middle'], df['Bollinger_Lower'] = ta.BBANDS(closes)
        df['EMA_8'] = ta.EMA(closes, timeperiod=8)
        df['VWAP'] = (volumes * (highs + lows + closes) / 3).cumsum() / volumes.cumsum()
        df['OBV'] = ta.OBV(closes, volumes)
        
        return df.dropna()
        
    except Exception as e:
        print(f"Indicator error: {e}")
        return None

def calculate_fibonacci_levels(df, period=14):
    """Calculate Fibonacci retracement levels"""
    try:
        high = df['high'].rolling(window=period).max()
        low = df['low'].rolling(window=period).min()
        diff = high - low
        df['Fib_38_2'] = high - (diff * 0.382)
        df['Fib_61_8'] = high - (diff * 0.618)
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
        """New strategy method"""
        if all(col in df.columns for col in ['RSI', 'MACD', 'MACD_Signal', 'OBV']):
            df['Signal_RSI'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] > 65, -1, 0))
            df['Signal_MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
            df['Signal_OBV'] = np.where(df['OBV'] < df['RSI'], 1, 0)

            df['Buy_Strategy4'] = (df['Signal_RSI'] == 1) & (df['Signal_MACD'] == 1) & (df['Signal_OBV'] == 1)
            df['Sell_Strategy4'] = (df['Signal_RSI'] == -1) & (df['Signal_MACD'] == -1) & (df['Signal_OBV'] == 1)
        return df

def main():
    """Main loop with ML integration"""
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
                
                # ML Prediction for all symbols
                latest_row = df.iloc[-1]
                latest_data = [latest_row[feature] for feature in ml_model.features]
                
                # Get corresponding model file
                csv_file = f"{symbol}1.csv"
                prediction = ml_model.predict_trend(csv_file, latest_data)
                
                if prediction is not None:
                    trend = 'UPWARD' if prediction == 1 else 'DOWNWARD'
                    print(f"{symbol} - ML Prediction: {trend} Trend Detected")
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
