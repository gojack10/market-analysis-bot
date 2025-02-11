import pandas as pd
import numpy as np
import time
import requests
import pygame
import sys
from threading import Thread
import talib as ta

# Audio Configuration
THEME_SOUND = 'data/audio/themebot.mp3'
CALL_SOUND = 'data/audio/call_signal.mp3'
PUT_SOUND = 'data/audio/put_signal.mp3'

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
URL_TEMPLATE = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}"

def fetch_realtime_data(symbol):
    """Fetch real-time 5-minute intraday data for a given symbol from Alpha Vantage API"""
    try:
        url = URL_TEMPLATE.format(symbol=symbol, interval=INTERVAL, api_key=API_KEY)
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
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"Exception fetching data for {symbol}: {e}")
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

def monitor_strategies():
    """Main monitoring loop with continuous theme music"""
    # Start theme music in background thread
    Thread(target=play_theme_loop, daemon=True).start()
    
    engine = RealtimeStrategyEngine()
    
    while True:
        try:
            # [Keep all your existing data fetching and processing code here]
            
            # When signals occur, use play_signal_sound instead of beep
            print(f"\n=== Market Update @ {timestamp} ===")
            for strategy in [1, 2, 3]:
                try:
                    buy_signal = latest.get(f'Buy_Strategy{strategy}', False)
                    sell_signal = latest.get(f'Sell_Strategy{strategy}', False)
                    
                    if buy_signal:
                        print(f"STRATEGY {strategy}: BUY CALLS")
                        play_signal_sound('call')
                    if sell_signal:
                        print(f"STRATEGY {strategy}: BUY PUTS")
                        play_signal_sound('put')
                except Exception as e:
                    print(f"Signal processing error ({strategy}): {e}")
            
            time.sleep(5)

        except KeyboardInterrupt:
            print("\nExiting...")
            pygame.mixer.quit()
            sys.exit(0)
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(10)

def main():
    """Main loop to fetch data and apply trading logic continuously for multiple symbols."""
    Thread(target=play_theme_loop, daemon=True).start()
    engine = RealtimeStrategyEngine()
    
    while True:
        for symbol in SYMBOLS:
            df = fetch_realtime_data(symbol)
            if df is not None and len(df) > 20:
                df.index = pd.to_datetime(df.index, utc=True)  # Ensure index is datetime and in UTC
                df = df.sort_index()  # Sort by datetime index
                
                latest_time = df.index[-1]  # Get the latest timestamp
                

                formatted_time = latest_time.strftime("%Y-%m-%d %H:%M:%S")  # Correct format
                
                print(f"\n=== Market Update @ {formatted_time} for {symbol} ===")  # Correct output
                
                df = engine.generate_signals_strategy1(df)
                df = engine.generate_signals_strategy2(df)
                df = engine.generate_signals_strategy3(df)
                
                for strategy in [1, 2, 3]:
                    buy_signal = df.iloc[-1].get(f'Buy_Strategy{strategy}', False)
                    sell_signal = df.iloc[-1].get(f'Sell_Strategy{strategy}', False)
                    
                    if buy_signal:
                        print(f"{symbol} - STRATEGY {strategy}: BUY CALLS")
                        play_signal_sound(symbol, 'call')
                    if sell_signal:
                        print(f"{symbol} - STRATEGY {strategy}: BUY PUTS")
                        play_signal_sound(symbol, 'put')
        
        time.sleep(5)  # Wait 5 seconds before fetching new data

if __name__ == "__main__":
    main()
