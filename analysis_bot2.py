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

# Audio Configuration
THEME_SOUND = 'data/audio/themebot.mp3'
CALL_SOUND = 'data/audio/CALLS.mp3'
PUT_SOUND = 'data/audio/PUTS.mp3'

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(2)

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
SYMBOLS = ['NVDA', 'AAPL']
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

def main():
    """Main loop with descriptive market update messages and corrected timezone handling"""
    Thread(target=play_theme_loop, daemon=True).start()
    engine = RealtimeStrategyEngine()
    
    while True:
        for symbol in SYMBOLS:
            df = fetch_realtime_data(symbol)
            if df is not None and len(df) > 20:
                # Ensure the DataFrame is sorted by time
                df = df.sort_index()

                # Calculate technical indicators and Fibonacci levels
                df = calculate_indicators(df)
                df = calculate_fibonacci_levels(df)
                
                # Get the latest data point
                latest = df.iloc[-1]
                timestamp = latest.name.strftime('%Y-%m-%d %H:%M:%S')
                
                # Print a more descriptive market update message
                print(f"\n=== Market Update for {symbol} ===")
                print(f"Timestamp: {timestamp}")
                print(f"Data points received: {len(df)}")
                print(f"Latest Close Price: {latest['close']:.2f}")
                if 'RSI' in latest:
                    print(f"Latest RSI: {latest['RSI']:.2f}")
                if 'MACD' in latest and 'MACD_Signal' in latest:
                    print(f"Latest MACD: {latest['MACD']:.4f} | MACD Signal: {latest['MACD_Signal']:.4f}")

                # Generate trading signals using three strategies
                df = engine.generate_signals_strategy1(df)
                df = engine.generate_signals_strategy2(df)
                df = engine.generate_signals_strategy3(df)
                
                # Check for buy or sell signals and play the corresponding audio
                for strategy in [1, 2, 3]:
                    buy_signal = df.iloc[-1].get(f'Buy_Strategy{strategy}', False)
                    sell_signal = df.iloc[-1].get(f'Sell_Strategy{strategy}', False)
                    
                    if buy_signal:
                        print(f"{symbol} - STRATEGY {strategy}: BUY CALLS")
                        play_signal_sound(symbol, 'call')
                    if sell_signal:
                        print(f"{symbol} - STRATEGY {strategy}: BUY PUTS")
                        play_signal_sound(symbol, 'put')
        
        time.sleep(5)

if __name__ == "__main__":
    main()
