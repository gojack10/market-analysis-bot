import numpy as np
import pandas as pd
import requests
import os
import talib  # Add TA-Lib for indicators
import time  # To create the interval for continuous checking
from datetime import datetime  # To check the market hours
from dotenv import load_dotenv
from collections import deque
from time import time, sleep
from termcolor import colored
import re

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()

    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit."""
        now = time()
        
        # Remove old requests outside the time window
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # If we've hit our limit, wait until enough time has passed
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.time_window - now
            if sleep_time > 0:
                print(f"[LOG] Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                sleep(sleep_time)
        
        # Add the new request timestamp
        self.requests.append(now)

class TradingBot:
    def __init__(self):
        load_dotenv()  # Load environment variables
        self.ticker = None
        self.data_daily = None
        self.data_hourly = None
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')  # Get API key from environment
        self.base_url = "https://www.alphavantage.co/query"
        self.market_open_time = "09:30"  # 9:30 AM (EST)
        self.market_close_time = "16:00"  # 4:00 PM (EST)
        self.macd_overbought_zone = None  # To store MACD overbought zone
        self.macd_oversold_zone = None   # To store MACD oversold zone
        self.swing_indicators = {}  # To store swing indicator states
        self.signals = []  # Initialize trade signals list
        # Initialize rate limiter: 10 requests per minute
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)

    def format_message(self, message):
        """Format 'bullish', 'bearish', 'call', 'calls', 'put', 'puts', 'buying', 'selling' with appropriate colors."""
    # Highlight 'bullish' in green
        message = re.sub(
            r'\b(bullish)\b',
            lambda m: colored(m.group(1), 'white', 'on_green', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
    # Highlight 'bearish' in red
        message = re.sub(
            r'\b(bearish)\b',
            lambda m: colored(m.group(1), 'white', 'on_red', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
    # Highlight 'call' and 'calls' in green
        message = re.sub(
            r'\b(call|calls)\b',
            lambda m: colored(m.group(1), 'white', 'on_green', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
    # Highlight 'put' and 'puts' in red
        message = re.sub(
            r'\b(put|puts)\b',
            lambda m: colored(m.group(1), 'white', 'on_red', attrs=['bold']),
            message,
            flags=re.IGNORECASE
    )
    # Highlight 'buying' in green
        message = re.sub(
            r'\b(buying)\b',
            lambda m: colored(m.group(1), 'white', 'on_green', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
    # Highlight 'selling' in red
        message = re.sub(
            r'\b(selling)\b',
            lambda m: colored(m.group(1), 'white', 'on_red', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
        return message

    def _make_request(self, url, params):
        """Make a rate-limited request to the API."""
        self.rate_limiter.wait_if_needed()
        response = requests.get(url, params=params)
        
        # Check for rate limit messages in the response
        if response.status_code == 200:
            data = response.json()
            if "Note" in data:
                print("[WARN] API call frequency note received:", data["Note"])
            if "Information" in data:
                print("[INFO] API information:", data["Information"])
        
        return response

    def _make_request(self, url, params):
        """Make a rate-limited request to the API."""
        self.rate_limiter.wait_if_needed()
        response = requests.get(url, params=params)
        
        # Check for rate limit messages in the response
        if response.status_code == 200:
            data = response.json()
            if "Note" in data:
                print("[WARN] API call frequency note received:", data["Note"])
            if "Information" in data:
                print("[INFO] API information:", data["Information"])
        
        return response

    def validate_key(self):
        """Validate the API key by making a simple request."""
        print("[LOG] Validating API key...")
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "IBM",  # Using IBM as a test symbol
            "apikey": self.api_key
        }
        response = self._make_request(self.base_url, params)
        if response.status_code == 200 and "Error Message" not in response.json():
            print("[LOG] API key is valid.")
        else:
            print(f"[ERROR] Invalid API key: {response.status_code}, {response.text}")
            raise ValueError("Invalid API key. Please check and try again.")

    def fetch_data(self, interval):
        """Fetch historical data for the given ticker and interval."""
        print(f"[LOG] Fetching {interval} data for ticker {self.ticker}...")
        
        # Map intervals to Alpha Vantage function names
        interval_mapping = {
            "daily": "TIME_SERIES_DAILY",
            "hourly": "TIME_SERIES_INTRADAY",
            "weekly": "TIME_SERIES_WEEKLY",
            "monthly": "TIME_SERIES_MONTHLY"
        }
        
        params = {
            "function": interval_mapping.get(interval, "TIME_SERIES_DAILY"),
            "symbol": self.ticker,
            "apikey": self.api_key,
            "outputsize": "full"  # Get full data
        }
        
        # Add interval parameter for intraday data
        if interval == "hourly":
            params["interval"] = "60min"
        
        response = self._make_request(self.base_url, params)

        if response.status_code == 200:
            data = response.json()
            
            # Handle different response formats based on interval
            time_series_key = {
                "daily": "Time Series (Daily)",
                "hourly": "Time Series (60min)",
                "weekly": "Weekly Time Series",
                "monthly": "Monthly Time Series"
            }.get(interval)
            
            if time_series_key not in data:
                print(f"[ERROR] No {interval} data received.")
                if "Error Message" in data:
                    print(f"[ERROR] API Error: {data['Error Message']}")
                elif "Note" in data:
                    print(f"[WARN] API Note: {data['Note']}")
                return None
                
            # Convert the nested dictionary to a DataFrame
            historical_data = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            
            # Rename columns to match our existing code
            historical_data.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }, inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                historical_data[col] = historical_data[col].astype(float)
            
            # Sort index in ascending order (oldest to newest)
            historical_data.sort_index(inplace=True)
            
            return historical_data
        else:
            print(f"[ERROR] Failed to fetch historical data: {response.status_code}, {response.text}")
            return None

    def determine_market_trend(self, data):
        """Determine the overall market trend using SMA, EMA, RSI, MACD."""
    
        # Check if we have enough data points
        if data is None or len(data) < 50:  # Adjusted minimum length to 50 for daily data
            print("[ERROR] Not enough data to calculate trends.")
            return "Insufficient data"

        # Simple and Exponential Moving Averages
        data['SMA50'] = data['close'].rolling(window=50).mean()
        data['SMA200'] = data['close'].rolling(window=200).mean()
        data['EMA50'] = data['close'].ewm(span=50, adjust=False).mean()
        data['EMA200'] = data['close'].ewm(span=200, adjust=False).mean()

        # RSI - Relative Strength Index
        data['RSI'] = talib.RSI(data['close'], timeperiod=21)

        # MACD - Moving Average Convergence Divergence
        macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal

        # Ensure that we have the last row's data for trend determination
        if len(data) > 0:
            latest = data.iloc[-1]

            # Trend Logic
            if latest['SMA50'] > latest['SMA200'] and latest['EMA50'] > latest['EMA200'] and latest['RSI'] < 70 and latest['MACD'] > latest['MACD_Signal']:
                return "Uptrend"
            elif latest['SMA50'] < latest['SMA200'] and latest['EMA50'] < latest['EMA200'] and latest['RSI'] > 30 and latest['MACD'] < latest['MACD_Signal']:
                return "Downtrend"
            else:
                return "Sideways"
        else:
            return "No trend data available"

    def fetch_fundamental_data(self):
        """Fetch fundamental data using Alpha Vantage's OVERVIEW endpoint."""
        print(f"[LOG] Fetching fundamental data for {self.ticker}...")
        params = {
            "function": "OVERVIEW",
            "symbol": self.ticker,
            "apikey": self.api_key
        }
        response = self._make_request(self.base_url, params)
        if response.status_code == 200:
            data = response.json()
            if "Error Message" in data:
                print(f"[ERROR] Failed to fetch fundamentals: {data['Error Message']}")
                return None
            return data
        print(f"[ERROR] Failed to fetch fundamentals: {response.status_code}, {response.text}")
        return None

    def calculate_rvi(self, data, period=10):
        """Calculate the Relative Vigor Index (RVI)."""
        close_open = data['close'] - data['open']
        high_low = data['high'] - data['low']
    
        rvi = talib.SMA(close_open, timeperiod=period) / talib.SMA(high_low, timeperiod=period)
        signal = talib.SMA(rvi, timeperiod=period)

        return {'RVI': rvi.iloc[-1], 'Signal': signal.iloc[-1]}

    def calculate_coppock_curve(self, data):
        """Calculate the Coppock Curve for long-term trend reversals."""
        roc_14 = talib.ROC(data['close'], timeperiod=14)
        roc_11 = talib.ROC(data['close'], timeperiod=11)
        coppock = talib.WMA(roc_14 + roc_11, timeperiod=10)

        return {'Coppock Curve': coppock.iloc[-1]}

    def calculate_dpo(self, data, period=20):
        """Calculate the Detrended Price Oscillator (DPO)."""
        sma = talib.SMA(data['close'], timeperiod=period)
        dpo = data['close'].shift(int(period / 2) + 1) - sma

        return {'DPO': dpo.iloc[-1]}

    def calculate_kvo(self, data):
        """Calculate the Klinger Volume Oscillator (KVO)."""
        adl = ((2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'])) * data['volume']
        short_ema = talib.EMA(adl, timeperiod=34)
        long_ema = talib.EMA(adl, timeperiod=55)
        kvo = short_ema - long_ema

        return {'KVO': kvo.iloc[-1]}


    def calculate_adx_dmi(self, data, period=14):
        """Calculate ADX and DMI indicators."""
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=period)
        plus_di = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
        minus_di = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
        
        latest_adx = adx.iloc[-1]
        latest_plus_di = plus_di.iloc[-1]
        latest_minus_di = minus_di.iloc[-1]
        
        trend_signal = "No strong trend"
        if latest_adx > 25:
            if latest_plus_di > latest_minus_di:
                trend_signal = "Strong Bullish Trend"
            elif latest_minus_di > latest_plus_di:
                trend_signal = "Strong Bearish Trend"
        
        return {
            'ADX': latest_adx,
            'Plus_DI': latest_plus_di,
            'Minus_DI': latest_minus_di,
            'Trend Signal': trend_signal
        }

    def calculate_mag7_moving_averages(self, data):
        """
        Mag7-specific moving average strategy optimized for daily charts of
        Magnificent 7 stocks (AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA)
        """
        mag7_tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']
        
        if self.ticker not in mag7_tickers:
            return None

        # Mag7-specific parameters
        data['SMA20'] = data['close'].rolling(window=20).mean()
        data['SMA50'] = data['close'].rolling(window=50).mean()
        
        # Get latest values
        current_price = data['close'].iloc[-1]
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]
        
        # Generate signal
        if current_price > sma20 and current_price > sma50:
            signal = "BULLISH"
            reason = "Price above both SMA20 and SMA50"
        elif current_price < sma20 and current_price < sma50:
            signal = "BEARISH"
            reason = "Price below both SMA20 and SMA50"
        else:
            signal = "NEUTRAL"
            reason = "Price between SMA20 and SMA50"
            
        return {
            'SMA20': sma20,
            'SMA50': sma50,
            'signal': signal,
            'reason': reason
        }

    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price (VWAP)."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        cumulative_tpv = (typical_price * data['volume']).cumsum()
        cumulative_vol = data['volume'].cumsum()
        vwap = cumulative_tpv / cumulative_vol
        return vwap

    def calculate_yearly_fibonacci_retracement(self, data, lookback_period=73*5):
        """Calculate Fibonacci retracement levels based on recent swing points."""
    
    # Select recent data within the lookback period
        recent_data = data.tail(lookback_period)
    
    # Find recent swing high and low
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
    
        diff = swing_high - swing_low  # Price range
    
    # Determine trend direction (assuming if last close is near high, it's an uptrend)
        last_close = recent_data['close'].iloc[-1]
        uptrend = abs(last_close - swing_high) < abs(last_close - swing_low)
    
        if uptrend:
        # Calculate retracement levels from low to high (support levels)
            levels = {
                '0.0%': swing_low,
                '23.6%': swing_low + 0.236 * diff,
                '38.2%': swing_low + 0.382 * diff,
                '50.0%': swing_low + 0.5 * diff,
                '61.8%': swing_low + 0.618 * diff,
                '100.0%': swing_high
            }
        else:
        # Calculate retracement levels from high to low (resistance levels)
            levels = {
                '100.0%': swing_high,
                '61.8%': swing_high - 0.618 * diff,
                '50.0%': swing_high - 0.5 * diff,
                '38.2%': swing_high - 0.382 * diff,
                '23.6%': swing_high - 0.236 * diff,
            '   0.0%': swing_low
            }

        return levels

    def calculate_90days_fibonacci_retracement(self, data, lookback_period=30*3):
        """Calculate Fibonacci retracement levels based on recent swing points."""
    
    # Select recent data within the lookback period
        recent_data = data.tail(lookback_period)
    
    # Find recent swing high and low
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
    
        diff = swing_high - swing_low  # Price range
    
    # Determine trend direction (assuming if last close is near high, it's an uptrend)
        last_close = recent_data['close'].iloc[-1]
        uptrend = abs(last_close - swing_high) < abs(last_close - swing_low)
    
        if uptrend:
        # Calculate retracement levels from low to high (support levels)
            levels = {
                '0.0%': swing_low,
                '23.6%': swing_low + 0.236 * diff,
                '38.2%': swing_low + 0.382 * diff,
                '50.0%': swing_low + 0.5 * diff,
                '61.8%': swing_low + 0.618 * diff,
                '100.0%': swing_high
            }
        else:
        # Calculate retracement levels from high to low (resistance levels)
            levels = {
                '100.0%': swing_high,
                '61.8%': swing_high - 0.618 * diff,
                '50.0%': swing_high - 0.5 * diff,
                '38.2%': swing_high - 0.382 * diff,
                '23.6%': swing_high - 0.236 * diff,
            '   0.0%': swing_low
            }

        return levels

    def calculate_stochastic_oscillator(self, data):
        """Calculate the Stochastic Oscillator and analyze its signals."""
        data['%K'], data['%D'] = talib.STOCH(data['high'], data['low'], data['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        
        latest_k = data['%K'].iloc[-1]
        latest_d = data['%D'].iloc[-1]
        latest_close = data['close'].iloc[-1]

        # Identify divergence for potential reversals/continuations
        if len(data) >= 20:
            prev_k = data['%K'].iloc[-2]
            prev_close = data['close'].iloc[-2]
            
            # Regular divergence (reversal)
            if (latest_close > prev_close and latest_k < prev_k):
                print(self.format_message("[MARKET SENTIMENT] Regular bearish divergence detected. Potential reversal down."))
            elif (latest_close < prev_close and latest_k > prev_k):
                print(self.format_message("[MARKET SENTIMENT] Regular bullish divergence detected. Potential reversal up."))
            
            # Hidden divergence (continuation)
            if (latest_close > prev_close and latest_k > prev_k):
                print(self.format_message("[MARKET SENTIMENT] Hidden bullish divergence detected. Potential continuation up."))
            elif (latest_close < prev_close and latest_k < prev_k):
                print(self.format_message("[MARKET SENTIMENT] Hidden bearish divergence detected. Potential continuation down."))

        # Determine trade decision
        if latest_k < 20 and latest_d < 20:
            print(self.format_message("[MARKET SENTIMENT] The stochastic Oscillator suggests oversold conditions, bullish signal."))
            return "CALL"
        elif latest_k > 80 and latest_d > 80:
            print(self.format_message("[MARKET SENTIMENT] The stochastic Oscillator suggests overbought conditions, bearish signal."))
            return "PUT"
        else:
            print("[TRADE IDEA] The stochastic Oscillator is neutral, no strong signal.")
            return "Neutral"

    def calculate_bollinger_bands(self, data, window=30):
        """Calculate Bollinger Bands."""
        # Calculate moving average and standard deviation
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()

        # Calculate the upper and lower Bollinger Bands
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        return upper_band, lower_band

    def rsi_trade_decision(self, data):
        """Make trade decisions based on 14-period RSI values."""
        # Get the latest 14-period RSI value
        latest_rsi_14 = data['RSI_14'].iloc[-1]
        
        # Trade decision based on RSI
        if latest_rsi_14 > 70:
            print(self.format_message("[MARKET SENTIMENT] 14-period RSI indicates overbought conditions, bearish signal."))
        elif latest_rsi_14 < 30:
            print(self.format_message("[MARKET SENTIMENT] 14-period RSI indicates oversold conditions, bullish signal."))
        else:
            print("[MARKET SENTIMENT] 14-period RSI is neutral, no strong signal.")
        
        # Return suggestion
        if latest_rsi_14 > 70:
            return "Bearish"
        elif latest_rsi_14 < 30:
            return "Bullish"
        else:
            return "Neutral"

    def calculate_williams_fractals(self, data):
        """Calculate the Williams Fractals indicator."""
        fractals = pd.Series(index=data.index)
        for i in range(2, len(data) - 2):
            if data['high'].iloc[i] > max(data['high'].iloc[i - 2], data['high'].iloc[i + 2]) and \
               data['high'].iloc[i] > data['high'].iloc[i - 1] and data['high'].iloc[i] > data['high'].iloc[i + 1]:
                fractals.iloc[i] = data['high'].iloc[i]  # Bullish fractal
            elif data['low'].iloc[i] < min(data['low'].iloc[i - 2], data['low'].iloc[i + 2]) and \
                 data['low'].iloc[i] < data['low'].iloc[i - 1] and data['low'].iloc[i] < data['low'].iloc[i + 1]:
                fractals.iloc[i] = data['low'].iloc[i]  # Bearish fractal
        return fractals.dropna()

    def calculate_atr(self, data, period=14):
        """Calculate the Average True Range (ATR) and generate stop-loss and take-profit levels for calls and puts."""
        data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
        latest_atr = data['ATR'].iloc[-1]
        current_price = data['close'].iloc[-1]

    # Calculate stop-loss and take-profit levels for long and short positions
        stop_loss_long = current_price - latest_atr
        take_profit_long = current_price + latest_atr
        stop_loss_short = current_price + latest_atr
        take_profit_short = current_price - latest_atr


        return {
            "ATR": latest_atr,
            "stop_loss_long": stop_loss_long,
            "take_profit_long": take_profit_long,
            "stop_loss_short": stop_loss_short,
            "take_profit_short": take_profit_short,
        }

    def calculate_macd_zones(self, data):
        """
        Calculate MACD overbought and oversold zones based on historical MACD crossovers.
        Major crossovers are identified as significant changes in MACD values.
        """
        macd, macd_signal, _ = talib.MACD(data['close'], fastperiod=8, slowperiod=21, signalperiod=9)
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal

        # Identify major crossovers using .iloc for position-based indexing
        major_crossovers = []
        for i in range(1, len(data)):
            if (macd.iloc[i - 1] < macd_signal.iloc[i - 1] and macd.iloc[i] > macd_signal.iloc[i]) or \
               (macd.iloc[i - 1] > macd_signal.iloc[i - 1] and macd.iloc[i] < macd_signal.iloc[i]):
                crossover_value = macd.iloc[i]
                major_crossovers.append(crossover_value)

        # Define overbought and oversold zones based on historical crossovers
        if major_crossovers:
            self.macd_overbought_zone = np.mean(major_crossovers) + np.std(major_crossovers)
            self.macd_oversold_zone = np.mean(major_crossovers) - np.std(major_crossovers)
        else:
            self.macd_overbought_zone = None
            self.macd_oversold_zone = None

    def candlestick_patterns(self, data):
        """Analyze candlestick patterns to determine bullish, bearish, or neutral signals."""
        patterns = {
            talib.CDL2CROWS: 'Two Crows',
            talib.CDL3BLACKCROWS: 'Three Black Crows',
            talib.CDL3INSIDE: 'Three Inside Up/Down',
            talib.CDL3LINESTRIKE: 'Three-Line Strike',
            talib.CDL3OUTSIDE: 'Three Outside Up/Down',
            talib.CDL3STARSINSOUTH: 'Three Stars in the South',
            talib.CDL3WHITESOLDIERS: 'Three Advancing White Soldiers',
            talib.CDLABANDONEDBABY: 'Abandoned Baby',
            talib.CDLADVANCEBLOCK: 'Advance Block',
            talib.CDLBELTHOLD: 'Belt-hold',
            talib.CDLBREAKAWAY: 'Breakaway',
            talib.CDLCLOSINGMARUBOZU: 'Closing Marubozu',
            talib.CDLCONCEALBABYSWALL: 'Concealing Baby Swallow',
            talib.CDLCOUNTERATTACK: 'Counterattack',
            talib.CDLDARKCLOUDCOVER: 'Dark Cloud Cover',
            talib.CDLDOJI: 'Doji',
            talib.CDLDOJISTAR: 'Doji Star',
            talib.CDLDRAGONFLYDOJI: 'Dragonfly Doji',
            talib.CDLENGULFING: 'Engulfing',
            talib.CDLEVENINGDOJISTAR: 'Evening Doji Star',
            talib.CDLEVENINGSTAR: 'Evening Star',
            talib.CDLGAPSIDESIDEWHITE: 'Up/Down-gap side-by-side white lines',
            talib.CDLGRAVESTONEDOJI: 'Gravestone Doji',
            talib.CDLHAMMER: 'Hammer',
            talib.CDLHANGINGMAN: 'Hanging Man',
            talib.CDLHARAMI: 'Harami',
            talib.CDLHARAMICROSS: 'Harami Cross',
            talib.CDLHIGHWAVE: 'High-Wave Candle',
            talib.CDLHIKKAKE: 'Hikkake',
            talib.CDLHIKKAKEMOD: 'Modified Hikkake',
            talib.CDLHOMINGPIGEON: 'Homing Pigeon',
            talib.CDLIDENTICAL3CROWS: 'Identical Three Crows',
            talib.CDLINNECK: 'In-Neck',
            talib.CDLINVERTEDHAMMER: 'Inverted Hammer',
            talib.CDLKICKING: 'Kicking',
            talib.CDLKICKINGBYLENGTH: 'Kicking - bull/bear determined by the longer marubozu',
            talib.CDLLADDERBOTTOM: 'Ladder Bottom',
            talib.CDLLONGLEGGEDDOJI: 'Long-Legged Doji',
            talib.CDLLONGLINE: 'Long Line Candle',
            talib.CDLMARUBOZU: 'Marubozu',
            talib.CDLMATCHINGLOW: 'Matching Low',
            talib.CDLMATHOLD: 'Mat Hold',
            talib.CDLMORNINGDOJISTAR: 'Morning Doji Star',
            talib.CDLMORNINGSTAR: 'Morning Star',
            talib.CDLONNECK: 'On-Neck',
            talib.CDLPIERCING: 'Piercing',
            talib.CDLRICKSHAWMAN: 'Rickshaw Man',
            talib.CDLRISEFALL3METHODS: 'Rising/Falling Three Methods',
            talib.CDLSEPARATINGLINES: 'Separating Lines',
            talib.CDLSHOOTINGSTAR: 'Shooting Star',
            talib.CDLSHORTLINE: 'Short Line',
            talib.CDLSPINNINGTOP: 'Spinning Top',
            talib.CDLSTALLEDPATTERN: 'Stalled Pattern',
            talib.CDLSTICKSANDWICH: 'Stick Sandwich',
            talib.CDLTAKURI: 'Takuri',
            talib.CDLTASUKIGAP: 'Tasuki Gap',
            talib.CDLTHRUSTING: 'Thrusting',
            talib.CDLTRISTAR: 'Tristar',
            talib.CDLUNIQUE3RIVER: 'Unique 3 River',
            talib.CDLUPSIDEGAP2CROWS: 'Upside Gap Two Crows',
            talib.CDLXSIDEGAP3METHODS: 'Upside/Downside Gap Three Methods'
        }

    # Check for the latest candlestick pattern (last candle in the data)
        pattern_signals = {}
        for pattern_func, pattern_name in patterns.items():
            pattern_result = pattern_func(data['open'], data['high'], data['low'], data['close'])
            if pattern_result.iloc[-1] != 0:
                pattern_signals[pattern_name] = pattern_result.iloc[-1]

        return pattern_signals

    def ichimoku_cloud(self, data):
        """Ichimoku Cloud system for swing trading"""
        tenkan_period = 9
        kijun_period = 26
        senkou_span_b_period = 52
        
        tenkan_high = data['high'].rolling(tenkan_period).max()
        tenkan_low = data['low'].rolling(tenkan_period).min()
        data['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

        kijun_high = data['high'].rolling(kijun_period).max()
        kijun_low = data['low'].rolling(kijun_period).min()
        data['kijun_sen'] = (kijun_high + kijun_low) / 2

        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        
        senkou_b_high = data['high'].rolling(senkou_span_b_period).max().shift(26)
        senkou_b_low = data['low'].rolling(senkou_span_b_period).min().shift(26)
        data['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(26)

        # Generate signals
        current_price = data['close'].iloc[-1]
        above_cloud = current_price > data['senkou_span_a'].iloc[-1] and current_price > data['senkou_span_b'].iloc[-1]
        below_cloud = current_price < data['senkou_span_a'].iloc[-1] and current_price < data['senkou_span_b'].iloc[-1]
        
        signal = ""
        if data['tenkan_sen'].iloc[-1] > data['kijun_sen'].iloc[-1] and above_cloud:
            signal = "CALL"
        elif data['tenkan_sen'].iloc[-1] < data['kijun_sen'].iloc[-1] and below_cloud:
            signal = "PUT"
        
        return {
            'signal': signal,
            'tenkan': data['tenkan_sen'].iloc[-1],
            'kijun': data['kijun_sen'].iloc[-1],
            'cloud_top': max(data['senkou_span_a'].iloc[-1], data['senkou_span_b'].iloc[-1]),
            'cloud_bottom': min(data['senkou_span_a'].iloc[-1], data['senkou_span_b'].iloc[-1])
        }

    def volume_weighted_macd(self, data):
        """Volume Weighted MACD for swing trading"""
        volume = data['volume'].astype(float)
        vwma_fast = (data['close'] * volume).rolling(12).sum() / volume.rolling(12).sum()
        vwma_slow = (data['close'] * volume).rolling(26).sum() / volume.rolling(26).sum()
        vwmacd = vwma_fast - vwma_slow
        signal_line = vwmacd.rolling(9).mean()
        
        crossover_up = vwmacd.iloc[-1] > signal_line.iloc[-1] and vwmacd.iloc[-2] <= signal_line.iloc[-2]
        crossover_down = vwmacd.iloc[-1] < signal_line.iloc[-1] and vwmacd.iloc[-2] >= signal_line.iloc[-2]
        
        trade_signal = ""
        if crossover_up:
            trade_signal = "CALL"
        elif crossover_down:
            trade_signal = "PUT"
            
        return {'signal': trade_signal, 'vwmacd': vwmacd.iloc[-1], 'signal_line': signal_line.iloc[-1]}

    def supertrend(self, data, period=10, multiplier=3):
        """Supertrend indicator for trend following"""
        hl2 = (data['high'] + data['low']) / 2
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=data.index)
        direction = pd.Series(1, index=data.index)

        for i in range(1, len(data)):
            if data['close'].iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif data['close'].iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]

        current_signal = "CALL" if direction.iloc[-1] == 1 else "PUT"
        return {'signal': current_signal, 'supertrend': supertrend.iloc[-1]}

    def elders_force_index(self, data, period=13):
        """Elder's Force Index for detecting buying/selling pressure"""
        fi = (data['close'].diff(period) * data['volume']).rolling(period).mean()
        ema_fi = fi.ewm(span=period).mean()
        
        signal = ""
        if ema_fi.iloc[-1] > 0 and ema_fi.iloc[-1] > ema_fi.iloc[-2]:
            signal = "CALL"
        elif ema_fi.iloc[-1] < 0 and ema_fi.iloc[-1] < ema_fi.iloc[-2]:
            signal = "PUT"
            
        return {'signal': signal, 'fi': ema_fi.iloc[-1]}

    def money_flow_index(self, data, period=14):
        """Money Flow Index for overbought/oversold conditions"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        raw_money_flow = typical_price * data['volume']
        
        positive_flow = raw_money_flow.where(data['close'] > data['close'].shift(1), 0)
        negative_flow = raw_money_flow.where(data['close'] < data['close'].shift(1), 0)
        
        ratio = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()
        mfi = 100 - (100 / (1 + ratio))
        
        signal = ""
        if mfi.iloc[-1] > 80:
            signal = "PUT"
        elif mfi.iloc[-1] < 20:
            signal = "CALL"
            
        return {'signal': signal, 'mfi': mfi.iloc[-1]}

    def chaikin_oscillator(self, data):
        """Chaikin Oscillator for accumulation/distribution"""
        adl = ((2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'])) * data['volume']
        adl = adl.cumsum()
        chaikin = adl.ewm(span=3).mean() - adl.ewm(span=10).mean()
        
        signal = ""
        if chaikin.iloc[-1] > 0 and chaikin.iloc[-2] <= 0:
            signal = "CALL"
        elif chaikin.iloc[-1] < 0 and chaikin.iloc[-2] >= 0:
            signal = "PUT"
            
        return {'signal': signal, 'chaikin': chaikin.iloc[-1]}

    def vortex_indicator(self, data, period=14):
        """Vortex Indicator for trend direction"""
    # Calculate True Range using TA-Lib
        data['true_range'] = talib.TRANGE(data['high'], data['low'], data['close'])
    
        vm_plus = abs(data['high'] - data['low'].shift(1))
        vm_minus = abs(data['low'] - data['high'].shift(1))
    
        vi_plus = vm_plus.rolling(period).sum() / data['true_range'].rolling(period).sum()
        vi_minus = vm_minus.rolling(period).sum() / data['true_range'].rolling(period).sum()
    
        signal = ""
        if vi_plus.iloc[-1] > vi_minus.iloc[-1] and vi_plus.iloc[-1] > 1:
            signal = "CALL"
        elif vi_minus.iloc[-1] > vi_plus.iloc[-1] and vi_minus.iloc[-1] > 1:
            signal = "PUT"
        
        return {'signal': signal, 'vi_plus': vi_plus.iloc[-1], 'vi_minus': vi_minus.iloc[-1]}

    def keltner_channels(self, data, period=20, multiplier=2):
        """Keltner Channels for volatility-based signals"""
        ema = data['close'].ewm(span=period).mean()
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        signal = ""
        if data['close'].iloc[-1] > upper.iloc[-1]:
            signal = "PUT"
        elif data['close'].iloc[-1] < lower.iloc[-1]:
            signal = "CALL"
            
        return {'signal': signal, 'upper': upper.iloc[-1], 'lower': lower.iloc[-1]}

    def donchian_channels(self, data, period=20):
        """Donchian Channels for breakout signals"""
        upper = data['high'].rolling(period).max()
        lower = data['low'].rolling(period).min()
        
        signal = ""
        if data['close'].iloc[-1] > upper.iloc[-2]:
            signal = "CALL"
        elif data['close'].iloc[-1] < lower.iloc[-2]:
            signal = "PUT"
            
        return {'signal': signal, 'upper': upper.iloc[-1], 'lower': lower.iloc[-1]}

    def rate_of_change(self, data, period=14):
        """Rate of Change for momentum detection"""
        roc = ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100
        ema_roc = roc.ewm(span=9).mean()
        
        signal = ""
        if roc.iloc[-1] > 0 and ema_roc.iloc[-1] > 0:
            signal = "CALL"
        elif roc.iloc[-1] < 0 and ema_roc.iloc[-1] < 0:
            signal = "PUT"
            
        return {'signal': signal, 'roc': roc.iloc[-1], 'ema_roc': ema_roc.iloc[-1]}

    def schaff_trend_cycle(self, data, period=23):
        """Schaff Trend Cycle for cyclical trends"""
        macd = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
        stc = 100 * (macd - macd.rolling(period).min()) / (macd.rolling(period).max() - macd.rolling(period).min())
        
        signal = ""
        if stc.iloc[-1] > 75:
            signal = "PUT"
        elif stc.iloc[-1] < 25:
            signal = "CALL"
            
        return {'signal': signal, 'stc': stc.iloc[-1]}

    def detect_volume_spikes(self, data, window=20):
        """Detect unusual volume activity for potential breakouts"""
        avg_volume = data['volume'].rolling(window).mean()
        current_volume = data['volume'].iloc[-1]
        
        signal = ""
        if current_volume > 2.5 * avg_volume.iloc[-1]:
            if data['close'].iloc[-1] > data['open'].iloc[-1]:
                signal = "CALL"
            else:
                signal = "PUT"
                
        return {'signal': signal, 'volume_ratio': current_volume / avg_volume.iloc[-1]}


    def calculate_support_resistance(self, data, lookback_periods=[30, 60, 90]):
        """
        Calculate support and resistance levels using recent highs and lows for multiple lookback periods.
        """
        support_resistance_levels = {}

        for period in lookback_periods:
        # Calculate recent highs and lows
            recent_high = data['high'].rolling(window=period).max().iloc[-1]
            recent_low = data['low'].rolling(window=period).min().iloc[-1]

        # Calculate support and resistance levels
            support_level = recent_low
            resistance_level = recent_high

            support_resistance_levels[period] = (support_level, resistance_level)

        return support_resistance_levels

    def log_fundamentals(self, data):
        """Log fundamental metrics in a readable format."""
        if not data:
            return
        
        # Helper function to format numbers
        def format_number(value, is_percent=False):
            try:
                num = float(value)
                if is_percent:
                    return f"{num:.2f}%"
                if num >= 1e12:
                    return f"${num/1e12:.2f}T"
                if num >= 1e9:
                    return f"${num/1e9:.2f}B"
                if num >= 1e6:
                    return f"${num/1e6:.2f}M"
                return f"${num:,.2f}"
            except:
                return "N/A"

        # Market Capitalization
        mcap = data.get('MarketCapitalization')
        print(f"[MARKET DATA] Market Cap: {format_number(mcap)}")

        # Valuation Metrics
        pe_ttm = data.get('[MARKET DATA] PERatio', 'N/A')
        pe_fwd = data.get('[MARKET DATA] ForwardPE', 'N/A')
        pb = data.get('[MARKET DATA] PriceToBookRatio', 'N/A')
        ps = data.get('[MARKET DATA] PriceToSalesRatioTTM', 'N/A')
        
        print(f"[MARKET DATA] P/E (TTM): {pe_ttm}")
        print(f"[MARKET DATA] P/E (FWD): {pe_fwd}")
        print(f"[MARKET DATA] P/B: {pb}")
        print(f"[MARKET DATA] P/S: {ps}")

        # Profitability Metrics
        eps = data.get('[MARKET DATA] EPS', 'N/A')
        roe = data.get('[MARKET DATA] ReturnOnEquityTTM', 'N/A')
        print(f"[MARKET DATA] EPS (TTM): ${eps}")
        print(f"[MARKET DATA] ROE (TTM): {format_number(roe, is_percent=True)}")

        # Liquidity and Efficiency
        current_ratio = data.get('[MARKET DATA] CurrentRatio', 'N/A')
        profit_margin = data.get('[MARKET DATA] ProfitMargin', 'N/A')
        print(f"[MARKET DATA] Current Ratio: {current_ratio}")
        print(f"[MARKET DATA] Profit Margin: {format_number(profit_margin, is_percent=True)}")

        # Growth Metrics
        revenue_growth = data.get('[MARKET DATA] QuarterlyRevenueGrowthYOY', 'N/A')
        eps_growth = data.get('[MARKET DATA] QuarterlyEarningsGrowthYOY', 'N/A')
        print(f"[MARKET DATA] Revenue Growth (YoY): {format_number(revenue_growth, is_percent=True)}")
        print(f"[MARKET DATA] EPS Growth (YoY): {format_number(eps_growth, is_percent=True)}")

        # Dividend Information
        dividend_yield = data.get('[MARKET DATA] DividendYield', 'N/A')
        print(f"[MARKET DATA] Dividend Yield: {format_number(dividend_yield, is_percent=True)}")

        # Technical Metrics
        beta = data.get('Beta', 'N/A')
        high_52 = data.get('52WeekHigh', 'N/A')
        low_52 = data.get('52WeekLow', 'N/A')
        print(f"[MARKET DATA] 52-Week Range: {format_number(low_52)} - {format_number(high_52)}")
        print(f"[MARKET DATA] Beta: {beta}")

        # Additional Metrics
        shares_outstanding = data.get('SharesOutstanding', 'N/A')
        print(f"[MARKET DATA] Shares Outstanding: {format_number(shares_outstanding)}")

    def macd_trade_decision(self, data):
        """
        Generate trade signals based on MACD crossovers in overbought or oversold zones.
        """
        if self.macd_overbought_zone is None or self.macd_oversold_zone is None:
            print("[LOG] MACD zones not calculated. Skipping MACD trade decision.")
            return None

        volume = data['volume'].astype(float)
        vwma = (data['close'] * volume).cumsum() / volume.cumsum()
        macd, macd_signal, _ = talib.MACD(data['close'], fastperiod=8, slowperiod=21, signalperiod=5)
        latest_macd = macd.iloc[-1]
        latest_signal = macd_signal.iloc[-1]

    # Return the latest MACD value
        return latest_macd

    def calculate_1hour_fibonacci_retracement(self, data):
        """Calculate Fibonacci retracement levels based on recent 1-hour swing points."""
        recent_data = data.tail(24*5)  # Last 5 days of hourly data
        
        # Find swing high and low
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        diff = swing_high - swing_low

        # Determine trend direction
        last_close = recent_data['close'].iloc[-1]
        uptrend = last_close > recent_data['close'].iloc[-24]  # Compare to 24 periods ago (1 day)

        if uptrend:
            levels = {
                '23.6%': swing_high - 0.236 * diff,
                '38.2%': swing_high - 0.382 * diff,
                '50.0%': swing_high - 0.5 * diff,
                '61.8%': swing_high - 0.618 * diff,
                'swing_low': swing_low,
                'swing_high': swing_high
            }
        else:
            levels = {
                '23.6%': swing_low + 0.236 * diff,
                '38.2%': swing_low + 0.382 * diff,
                '50.0%': swing_low + 0.5 * diff,
                '61.8%': swing_low + 0.618 * diff,
                'swing_low': swing_low,
                'swing_high': swing_high
            }
        return levels, uptrend

    def fib_strategy_1hour(self, data):
        """Implement Fibonacci Retracement Strategy for 1-hour chart"""
        current_price = data['close'].iloc[-1]
        fib_levels, trend_direction = self.calculate_1hour_fibonacci_retracement(data)
        
        # Calculate indicators
        data['EMA9'] = talib.EMA(data['close'], timeperiod=9)
        data['EMA21'] = talib.EMA(data['close'], timeperiod=21)
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        
        signals = []
        entry_level = None
        stop_loss = None
        profit_targets = []

        # Trend and Level Analysis
        if trend_direction:  # Uptrend - Look for Call entries
            for level in ['38.2%', '50.0%', '61.8%']:
                fib_value = fib_levels[level]
                if self._is_near_level(current_price, fib_value):
                    # Bullish confirmation checks
                    if (data['close'].iloc[-1] > data['open'].iloc[-1] and  # Bullish candle
                        data['RSI'].iloc[-1] > 40 and
                        data['EMA9'].iloc[-1] > data['EMA21'].iloc[-1]):
                        
                        entry_level = fib_value
                        stop_loss = fib_levels['swing_low'] - (0.01 * current_price)  # Buffer
                        profit_targets = [
                            fib_levels['23.6%'],
                            fib_levels['swing_high']
                        ]
                        signals.append(self.format_message(
                            f"[1 HOUR FIB STRATEGY] CALL Entry at {level} level ({entry_level:.2f}) | "
                            f"SL: {stop_loss:.2f} | Targets: {profit_targets[0]:.2f}, {profit_targets[1]:.2f}"
                        ))
                        break

        else:  # Downtrend - Look for Put entries
            for level in ['38.2%', '50.0%', '61.8%']:
                fib_value = fib_levels[level]
                if self._is_near_level(current_price, fib_value):
                    # Bearish confirmation checks
                    if (data['close'].iloc[-1] < data['open'].iloc[-1] and  # Bearish candle
                        data['RSI'].iloc[-1] < 60 and
                        data['EMA9'].iloc[-1] < data['EMA21'].iloc[-1]):
                        
                        entry_level = fib_value
                        stop_loss = fib_levels['swing_high'] + (0.01 * current_price)  # Buffer
                        profit_targets = [
                            fib_levels['23.6%'],
                            fib_levels['swing_low']
                        ]
                        signals.append(self.format_message(
                            f"[1 HOUR FIB STRATEGY] PUT Entry at {level} level ({entry_level:.2f}) | "
                            f"SL: {stop_loss:.2f} | Targets: {profit_targets[0]:.2f}, {profit_targets[1]:.2f}"
                        ))
                        break

        # Log important levels
        print(self.format_message(
            f"[1 HOUR FIB STRATEGY] Key Levels | Swing High: {fib_levels['swing_high']:.2f} | "
            f"Swing Low: {fib_levels['swing_low']:.2f} | Trend: {'Uptrend' if trend_direction else 'Downtrend'}"
        ))
        for level, value in fib_levels.items():
            if '%' in level:
                print(f"[1 HOUR FIB STRATEGY] {level}: {value:.2f}")

        if not signals:
            signals.append("[1 HOUR FIB STRATEGY] No valid entries found at Fibonacci levels")
    
        return signals

    def generate_trade_signals(self, data):
        """Generate and log trade signals, ensuring proper order of logs."""
        trade_ideas = self.trade_idea(data)  # Generate trade ideas first
        fib_signals = self.fib_strategy_1hour(data)  # Generate Fibonacci signals second

        # Print trade ideas first
        if trade_ideas:
            for idea in trade_ideas:
                print(self.format_message(idea))
        else:
            print(self.format_message("[TRADE IDEA] No strong trade ideas generated based on the current analysis."))

        # Print Fibonacci strategy logs after trade ideas
        if fib_signals:
            for signal in fib_signals:
                print(self.format_message(signal))

    def _log_fib_levels(self, fib_levels):
        """Helper to log Fibonacci levels"""
        print(self.format_message(
            f"[1 HOUR FIB STRATEGY] Key Levels | Swing High: {fib_levels['swing_high']:.2f} | "
            f"Swing Low: {fib_levels['swing_low']:.2f}"
        ))
        for level, value in fib_levels.items():
            if '%' in level:
                print(f"[1 HOUR FIB STRATEGY] {level}: {value:.2f}")

    def trade_idea(self, data):
        """Determine a trade idea based on Fibonacci retracement, Bollinger Bands, RSI, MACD, and ATR."""
        self.analyze_historical_trends(data)
    # Initialize the trade_ideas list
        trade_ideas = []

        # Calculate all required indicators first
        data = self.calculate_indicators(data)
        
    # Calculate ADX DMI
        adx_dmi = self.calculate_adx_dmi(data)

    # Calculate RVI
        rvi_data = self.calculate_rvi(data)
        rvi, signal = rvi_data['RVI'], rvi_data['Signal']

    # Calculate Coppock
        coppock_data = self.calculate_coppock_curve(data)
        coppock = coppock_data['Coppock Curve']

    # Calculate DPO    
        dpo_data = self.calculate_dpo(data)
        dpo = dpo_data['DPO']

    # Calculate KVO
        kvo_data = self.calculate_kvo(data)
        kvo = kvo_data['KVO']

    # KVO logic
        if kvo > 0:
            print(self.format_message("[MARKET DATA] KVO indicates positive money flow. Potential buying pressure detected."))
            trade_ideas.append("[TRADE IDEA] KVO suggests institutional buying, consider call positions.")
        elif kvo < 0:
            print(self.format_message("[MARKET DATA] KVO indicates negative money flow. Potential selling pressure detected."))
            trade_ideas.append("[TRADE IDEA] KVO suggests institutional selling, consider put positions.")

    # Calculate Fibonacci retracement levels
        yearly_fib_levels = self.calculate_yearly_fibonacci_retracement(data)
        fib_levels_90days = self.calculate_90days_fibonacci_retracement(data)

    # Calculate Bollinger Bands
        upper_band, lower_band = self.calculate_bollinger_bands(data)

    # Calculate Support and Resistance for multiple periods
        support_resistance_levels = self.calculate_support_resistance(data)

    # Calculate MACD zones and trade decision
        self.calculate_macd_zones(data)
        latest_macd = self.macd_trade_decision(data)  # Get the latest MACD value

        print(f"[INDICATOR] ADX: {adx_dmi['ADX']:.2f}, +DI: {adx_dmi['Plus_DI']:.2f}, -DI: {adx_dmi['Minus_DI']:.2f}")

    # Calculate VWAP and log
        vwap = self.calculate_vwap(data)
        current_vwap = vwap.iloc[-1]
        print(f"[INDICATOR] Current VWAP: {current_vwap:.2f}")
    
    # Calculate ATR for stop loss and take profit levels
        atr_levels = self.calculate_atr(data)
        atr = atr_levels["ATR"]
        stop_loss_long = atr_levels["stop_loss_long"]
        take_profit_long = atr_levels["take_profit_long"]
        stop_loss_short = atr_levels["stop_loss_short"]
        take_profit_short = atr_levels["take_profit_short"]

        print(f"[INDICATOR] Current ATR: {atr}")

        print("[INDICATOR] Fibonacci Levels:")
        for level, price in yearly_fib_levels.items():
            print(f"{level}: {price}")
        print("[INDICATOR] Fibonacci Levels:")
        for level, price in fib_levels_90days.items():
            print(f"{level}: {price}")

        print("[INDICATOR] Bollinger Bands:")
        print(f"Upper Band: {upper_band.iloc[-1]}")
        print(f"Lower Band: {lower_band.iloc[-1]}")

        print("[INDICATOR] Support and Resistance Levels:")
        for period, levels in support_resistance_levels.items():
            print(f"{period} day Support Level: {levels[0]}")
            print(f"{period} day Resistance Level: {levels[1]}")

    # Add Mag7 Moving Average analysis
        mag7_analysis = self.calculate_mag7_moving_averages(data)
        if mag7_analysis:
            print(f"[INDICATOR] SMA20: {mag7_analysis['SMA20']:.2f}")
            print(f"[INDICATOR] SMA50: {mag7_analysis['SMA50']:.2f}")

    # Calculate additional RSI periods
        data['RSI_7'] = talib.RSI(data['close'], timeperiod=7)
        data['RSI_14'] = talib.RSI(data['close'], timeperiod=14)
        data['RSI_21'] = talib.RSI(data['close'], timeperiod=21)
    
        latest_rsi_7 = data['RSI_7'].iloc[-1]
        latest_rsi_21 = data['RSI_21'].iloc[-1]
    
    # Log all RSI values clearly
        print(f"[INDICATOR] 7-period RSI: {data['RSI_7'].iloc[-1]:.2f}")
        print(f"[INDICATOR] 14-period RSI: {data['RSI_14'].iloc[-1]:.2f}")
        print(f"[INDICATOR] 21-period RSI (Market Trend): {data['RSI'].iloc[-1]:.2f}")

    # Collect trade ideas
        trade_ideas = []
        current_price = data['close'].iloc[-1]
        print(f"[MARKET DATA] Current Price: {current_price}")

    # ADDED: Recent price and volume logging
        recent_high = data['high'].tail(30).max()  # 30-day high
        recent_low = data['low'].tail(30).min()    # 30-day low
        current_volume = data['volume'].iloc[-1]
    
        print(f"[MARKET DATA] 30-Day High: {recent_high:.2f}")
        print(f"[MARKET DATA] 30-Day Low: {recent_low:.2f}")
        print(f"[MARKET DATA] Current Volume: {current_volume:,}")  # Format with commas

    # Calculate and log volatility (ADDED SECTION)
        daily_returns = data['close'].pct_change().dropna()
        if len(daily_returns) >= 30:
            historical_volatility = daily_returns.tail(30).std()
            annualized_volatility = historical_volatility * np.sqrt(252)  # 252 trading days/year
            print(f"[MARKET DATA] 30-Day Historical Volatility (Daily): {historical_volatility:.4f}")
            print(f"[MARKET DATA] Annualized Volatility: {annualized_volatility:.4f}")
        else:
            print("[MARKET DATA] Not enough data to calculate 30-day historical volatility")

    # RVI Bullish
        if rvi > signal:
            print(self.format_message("[MARKET SENTIMENT] RVI indicates bullish momentum. Consider long positions."))
            trade_ideas.append("[TRADE IDEA] RVI crossover suggests buying opportunity.")
        elif rvi < signal:
            print(self.format_message("[MARKET SENTIMENT] RVI indicates bearish momentum. Consider short positions."))
            trade_ideas.append("[TRADE IDEA] RVI crossover suggests selling opportunity.")

    # Coppock Curve
        if coppock > 0:
            print(self.format_message("[MARKET SENTIMENT] Coppock Curve turning positive. Possible long-term bullish trend forming."))
            trade_ideas.append(self.format_message("[TRADE IDEA] Coppock Curve suggests long-term buying opportunity."))
        else:
            print(self.format_message("[MARKET SENTIMENT] Coppock Curve remains negative. No strong bullish trend yet."))

    # DPO
        if dpo > 0:
            print(self.format_message("[MARKET SENTIMENT] DPO is positive. Short-term bullish cycle detected."))
            trade_ideas.append("[TRADE IDEA] DPO suggests a buying opportunity for short-term swing trades.")
        elif dpo < 0:
            print(self.format_message("[MARKET SENTIMENT] DPO is negative. Short-term bearish cycle detected."))
            trade_ideas.append("[TRADE IDEA] DPO suggests a selling opportunity for short-term swing trades.")

    # KVO
        if kvo > 0:
            print(self.format_message("[MARKET SENTIMENT] KVO indicates positive money flow. Potential buying pressure detected."))
            trade_ideas.append("[TRADE IDEA] KVO suggests institutional buying, consider call positions.")
        elif kvo < 0:
            print(self.format_message("[MARKET SENTIMENT] KVO indicates negative money flow. Potential selling pressure detected."))
            trade_ideas.append("[TRADE IDEA] KVO suggests institutional selling, consider put positions.")

    # Ichimoku Cloud
        ichimoku = self.ichimoku_cloud(data)
        if ichimoku['signal']:
            trade_ideas.append(f"[TRADE IDEA] Ichimoku Cloud: {ichimoku['signal']} Signal (Price: {data['close'].iloc[-1]:.2f}, Cloud Top: {ichimoku['cloud_top']:.2f})")

    # Volume Weighted MACD
        vwmacd = self.volume_weighted_macd(data)
        if vwmacd['signal']:
            trade_ideas.append(f"[TRADE IDEA] Volume-Weighted MACD: {vwmacd['signal']} Signal (Value: {vwmacd['vwmacd']:.2f})")

    # Supertrend
        supertrend = self.supertrend(data)
        trade_ideas.append(f"[TRADE IDEA] Supertrend: {supertrend['signal']} (Current Level: {supertrend['supertrend']:.2f})")

    # Elder's Force Index
        fi = self.elders_force_index(data)
        if fi['signal']:
            trade_ideas.append(f"[TRADE IDEA] Force Index: {fi['signal']} (Value: {fi['fi']:.2f})")

    # Money Flow Index
        mfi = self.money_flow_index(data)
        if mfi['signal']:
            trade_ideas.append(f"[TRADE IDEA] Money Flow Index: {mfi['signal']} (Value: {mfi['mfi']:.2f})")

    # Chaikin Oscillator
        chaikin = self.chaikin_oscillator(data)
        if chaikin['signal']:
            trade_ideas.append(f"[TRADE IDEA] Chaikin Oscillator: {chaikin['signal']} (Value: {chaikin['chaikin']:.2f})")

    # Vortex Indicator
        vortex = self.vortex_indicator(data)
        if vortex['signal']:
            trade_ideas.append(f"[TRADE IDEA] Vortex: {vortex['signal']} (VI+: {vortex['vi_plus']:.2f}, VI-: {vortex['vi_minus']:.2f})")

    # Keltner Channels
        keltner = self.keltner_channels(data)
        if keltner['signal']:
            trade_ideas.append(f"[TRADE IDEA] Keltner: {keltner['signal']} (Price: {data['close'].iloc[-1]:.2f}, Upper: {keltner['upper']:.2f}, Lower: {keltner['lower']:.2f})")

    # Donchian Channels
        donchian = self.donchian_channels(data)
        if donchian['signal']:
            trade_ideas.append(f"[TRADE IDEA] Donchian Breakout: {donchian['signal']} (Price: {data['close'].iloc[-1]:.2f}, Upper: {donchian['upper']:.2f}, Lower: {donchian['lower']:.2f})")

    # Schaff Trend Cycle
        stc = self.schaff_trend_cycle(data)
        if stc['signal']:
            trade_ideas.append(f"[TRADE IDEA] Schaff Trend Cycle: {stc['signal']} (Value: {stc['stc']:.2f})")

    # Volume Spike Detection
        volume_spike = self.detect_volume_spikes(data)
        if volume_spike['signal']:
            trade_ideas.append(f"[TRADE IDEA] Volume Spike: {volume_spike['signal']} (Ratio: {volume_spike['volume_ratio']:.1f}x average)")

    # Rate of Change
        roc = self.rate_of_change(data)
        if roc['signal']:
            trade_ideas.append(f"[TRADE IDEA] Momentum ROC: {roc['signal']} (Value: {roc['roc']:.2f}%)")

    # Get SMA20 and SMA50 values
        sma20 = data['SMA20'].iloc[-1]
        sma50 = data['SMA50'].iloc[-1]

    # VWAP-based signals
        current_price = data['close'].iloc[-1]
    
    # Bullish conditions
        if current_price > current_vwap:
            trade_ideas.append("[TRADE IDEA] Price above VWAP â€” Look for bullish pullbacks.")
        # Check if VWAP is acting as support
            if data['low'].iloc[-1] <= current_vwap:
                trade_ideas.append("[TRADE IDEA] VWAP acting as support. Confirm with RSI/MACD.")
        # Check SMA alignment
            if current_price > sma20 and current_price > sma50:
                trade_ideas.append("[MARKET SENTIMENT] Bullish trend confirmed: Price above SMA20 & SMA50.")
    
    # Bearish conditions
        elif current_price < current_vwap:
            trade_ideas.append("[TRADE IDEA] Price below VWAP â€” Look for bearish rejections.")
        # Check if VWAP is acting as resistance
            if data['high'].iloc[-1] >= current_vwap:
                trade_ideas.append("[TRADE IDEA] VWAP acting as resistance. Confirm with RSI/MACD.")
        # Check SMA alignment
            if current_price < sma20 and current_price < sma50:
                trade_ideas.append("[MARKET SENTIMENT] Bearish trend confirmed: Price below SMA20 & SMA50.")

    # Bollinger Bands conditions
        if current_price > upper_band.iloc[-1]:
            trade_ideas.append("[MARKET SENTIMENT] Bearish, price above upper Bollinger Band.")
        elif current_price < lower_band.iloc[-1]:
            trade_ideas.append("[MARKET SENTIMENT] Bullish, price below lower Bollinger Band.")
        else:
            print("[MARKET SENTIMENT] Price is within Bollinger Bands. No strong signal.")

        if mag7_analysis:
            trade_ideas.append(
                f"[MARKET SENTIMENT] Moving Averages: {mag7_analysis['signal']} - {mag7_analysis['reason']}"
            )

    # Fibonacci retracement conditions
        if current_price > yearly_fib_levels['50.0%']:
            trade_ideas.append("[MARKET SENTIMENT] Bullish, price above 50% yearly Fibonacci retracement.")
        if yearly_fib_levels['38.2%'] < current_price <= yearly_fib_levels['50.0%']:
            trade_ideas.append("[MARKET SENTIMENT] Slightly bullish, price near 38.2% yearly Fibonacci retracement.")
        if yearly_fib_levels['23.6%'] < current_price <= yearly_fib_levels['38.2%']:
            trade_ideas.append("[MARKET SENTIMENT] Slightly bearish, price near 23.6% yearly Fibonacci retracement.")
        if current_price <= yearly_fib_levels['23.6%']:
            trade_ideas.append("[MARKET SENTIMENT] Bearish, price below 23.6% yearly Fibonacci retracement.")

        if current_price > fib_levels_90days['50.0%']:
            trade_ideas.append("[MARKET SENTIMENT] Bullish, price above 50% 90 days Fibonacci retracement.")
        if fib_levels_90days['38.2%'] < current_price <= fib_levels_90days['50.0%']:
            trade_ideas.append("[MARKET SENTIMENT] Slightly bullish, price near 38.2% 90 days Fibonacci retracement.")
        if fib_levels_90days['23.6%'] < current_price <= fib_levels_90days['38.2%']:
            trade_ideas.append("[MARKET SENTIMENT] Slightly bearish, price near 23.6% 90 days Fibonacci retracement.")
        if current_price <= fib_levels_90days['23.6%']:
            trade_ideas.append("[MARKET SENTIMENT] Bearish, price below 23.6% 90 days Fibonacci retracement.")

    # RSI-based trade decision
        rsi_signal = self.rsi_trade_decision(data)
        if rsi_signal == "Bullish":
            trade_ideas.append("[MARKET SENTIMENT] 14-period RSI indicates oversold conditions, suggesting a bullish opportunity.")
        elif rsi_signal == "Bearish":
            trade_ideas.append("[MARKET SENTIMENT] 14-period RSI indicates overbought conditions, suggesting a bearish opportunity.")

    # RSI 7 trade idea log
        if latest_rsi_7 > 70:
            print("[MARKET SENTIMENT] 7-period RSI indicates overbought conditions, bearish.")
        elif latest_rsi_7 < 30:
            print("[MARKET SENTIMENT] 7-period RSI indicates oversold conditions, bullish.")
        else:
            print("[MARKET SENTIMENT] 7-period RSI is neutral, no strong signal.")
    
    # RSI 21 trade idea log
        if latest_rsi_21 > 70:
            print("[MARKET SENTIMENT] 21-period RSI indicates overbought conditions, bearish.")
        elif latest_rsi_21 < 30:
            print("[MARKET SENTIMENT] 21-period RSI indicates oversold conditions, bullish.")
        else:
            print("[MARKET SENTIMENT] 21-period RSI is neutral, no strong signal.")
    
    # RSI convergence logic
        if latest_rsi_7 > latest_rsi_21:
            print("[MARKET SENTIMENT] 7-period RSI above 21-period RSI: Bullish momentum detected.")
        elif latest_rsi_7 < latest_rsi_21:
            print("[MARKET SENTIMENT] 7-period RSI below 21-period RSI: Bearish momentum detected.")
        else:
            print("[MARKET SENTIMENT] RSI 7 and 21 are equal: Neutral trend.")

    # Stochastic Oscillator decision
        stochastic_decision = self.calculate_stochastic_oscillator(data)
        if stochastic_decision == "CALL":
            trade_ideas.append("[TRADE IDEA] Stochastic Oscillator indicates CALL options.")
        elif stochastic_decision == "PUT":
            trade_ideas.append("[TRADE IDEA] Stochastic Oscillator indicates PUT options.")

    # MACD-based trade decision
        if latest_macd is not None and self.macd_overbought_zone is not None and self.macd_oversold_zone is not None:
            if latest_macd > self.macd_overbought_zone:
                trade_ideas.append("[TRADE IDEA] MACD is in overbought zone. Consider selling or buying PUT options.")
            elif latest_macd < self.macd_oversold_zone:
                trade_ideas.append("[TRADE IDEA] MACD is in oversold zone. Consider buying or buying CALL options.")
            else:
                print("[TRADE IDEA] MACD is in neutral zone. No strong signal.")

    # Candlestick pattern analysis
        pattern_signals = self.candlestick_patterns(data)
        for pattern, signal in pattern_signals.items():
            if signal > 0:
                trade_ideas.append(f"[TRADE IDEA] Bullish pattern detected: {pattern}")
            elif signal < 0:
                trade_ideas.append(f"[TRADE IDEA] Bearish pattern detected: {pattern}")

    # Support and Resistance analysis
        for period, levels in support_resistance_levels.items():
            if current_price > levels[1]:
                trade_ideas.append(f"[TRADE IDEA] Price broke above {period} day resistance, consider buying.")
            elif current_price < levels[0]:
                trade_ideas.append(f"[TRADE IDEA] Price broke below {period} day support, consider selling.")
            else:
                trade_ideas.append(f"[TRADE IDEA] Price is between {period} day support and resistance, no strong signal.")

    # Example conditions (retain and ensure all logs are captured)
        if current_price > data['high'].rolling(20).max().iloc[-2]:
            trade_ideas.append("[TRADE IDEA] Price breaking above 20-day high, consider buying.")
        if current_price < data['low'].rolling(20).min().iloc[-2]:
            trade_ideas.append("[TRADE IDEA] Price breaking below 20-day low, consider selling.")

# ... inside generate_trade_signals() method
        latest = data.iloc[-1]
        current_price = latest['close']
        atr = latest['ATR']

    # Long Call Recommendation (Bullish)
        entry_long_call = current_price - (atr * 0.5)  # Enter at a slight pullback to ATR support
        sl_long_call = entry_long_call - (atr * 1.5)
        tp_long_call = entry_long_call + (atr * 2)
        self.signals.append(
            f"[TRADE IDEA] BUY LONG CALL Enter at {entry_long_call:.2f} | "
            f"SL: {sl_long_call:.2f} | TP: {tp_long_call:.2f} | "
            f"Risk: {((entry_long_call - sl_long_call)/entry_long_call*100):.1f}% | "
            f"Reward: {((tp_long_call - entry_long_call)/entry_long_call*100):.1f}%"
        )

    # Long Put Recommendation (Bearish)
        entry_long_put = current_price + (atr * 0.5)  # Enter at a slight rally to ATR resistance
        sl_long_put = entry_long_put + (atr * 1.5)
        tp_long_put = entry_long_put - (atr * 2)
        self.signals.append(
            f"[TRADE IDEA] BUY LONG PUT Enter at {entry_long_put:.2f} | "
            f"SL: {sl_long_put:.2f} | TP: {tp_long_put:.2f} | "
            f"Risk: {((sl_long_put - entry_long_put)/entry_long_put*100):.1f}% | "
            f"Reward: {((entry_long_put - tp_long_put)/entry_long_put*100):.1f}%"
        )

# Log all recommendations
        log_file = "trade_signals.log"
        with open(log_file, "a") as f:
            for signal in self.signals:
                formatted_signal = self.format_message(signal)
                print(formatted_signal)  # Print to terminal
                f.write(formatted_signal + "\n")  # Save to file
        self.signals.clear()

        # Keep the existing return statement and final logging
        if trade_ideas:
            for idea in trade_ideas:
                print(self.format_message(idea))
        else:
            print(self.format_message("[TRADE IDEA] No strong trade ideas generated based on the current analysis."))

        return trade_ideas

    def _check_momentum_alignment(self, analysis):
        """Check if momentum indicators align across time frames."""
        aligned = []
        if analysis['10yr']['macd_trend'] == "Bullish": aligned.append("10yr")
        if analysis['1yr']['macd_trend'] == "Bullish": aligned.append("1yr")
        if analysis['3mo']['macd_trend'] == "Bullish": aligned.append("3mo")
        
        if len(aligned) == 3: return "Strong Bullish Confluence"
        if len(aligned) >= 2: return "Bullish Bias"
        if len(aligned) == 1: return "Mixed Signals"
        return "Bearish Dominance"

    def analyze_historical_trends(self, data):
        """Analyze price trends across different time horizons (10yr, 1yr, 3mo) and compare to historical data."""
        if len(data) < 252 * 10:  # Minimum 10 years of daily data
            print("[MARKET DATA] Warning: Insufficient data for full historical analysis")
            return None

        # Prepare time periods
        ten_year_data = data.copy()
        one_year_data = data.tail(252).copy()  # 252 trading days/year
        three_month_data = data.tail(63).copy()  # ~3 months (21 trading days/month)

        # Calculate key metrics for each period
        time_frames = {
            '10yr': ten_year_data,
            '1yr': one_year_data,
            '3mo': three_month_data
        }

        analysis = {}
        for timeframe, df in time_frames.items():
            # Calculate moving averages using .loc to avoid SettingWithCopyWarning
            df.loc[:, 'SMA200'] = df['close'].rolling(200).mean()
            df.loc[:, 'SMA50'] = df['close'].rolling(50).mean()
            
            # Current price position relative to averages
            current_price = df['close'].iloc[-1]
            sma200 = df['SMA200'].iloc[-1]
            sma50 = df['SMA50'].iloc[-1]
            
            # Momentum indicators
            rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
            macd, _, _ = talib.MACD(df['close'])
            
            analysis[timeframe] = {
                'price_vs_sma200': (current_price/sma200 - 1) * 100,
                'price_vs_sma50': (current_price/sma50 - 1) * 100,
                'sma50_vs_sma200': (sma50/sma200 - 1) * 100,
                'rsi': rsi,
                'macd_trend': "Bullish" if macd.iloc[-1] > macd.iloc[-5:-1].mean() else "Bearish",
                'current_price': current_price,
                'volatility': df['close'].pct_change().std() * np.sqrt(252)
            }

        # Generate trend comparison
        trend_summary = {
            'long_term_trend': "Bullish" if analysis['10yr']['price_vs_sma200'] > 0 else "Bearish",
            'medium_term_trend': "Bullish" if analysis['1yr']['price_vs_sma50'] > 0 else "Bearish",
            'short_term_trend': "Bullish" if analysis['3mo']['price_vs_sma50'] > 0 else "Bearish",
            'momentum_confluence': self._check_momentum_alignment(analysis)
        }

        # Print formatted analysis
        print("\n[MARKET DATA] Historical Price Patterns:")
        for timeframe in ['10yr', '1yr', '3mo']:
            print(f"\n--- {timeframe.upper()} ANALYSIS ---")
            print(f"Price vs 200 SMA: {analysis[timeframe]['price_vs_sma200']:.2f}%")
            print(f"Price vs 50 SMA: {analysis[timeframe]['price_vs_sma50']:.2f}%")
            print(f"RSI: {analysis[timeframe]['rsi']:.1f}")
            print(f"MACD Trend: {analysis[timeframe]['macd_trend']}")
            print(f"Annualized Volatility: {analysis[timeframe]['volatility']:.2%}")

        print("\n[MARKET DATA] Trend Summary:")
        print(f"Long-term (10yr) Trend: {trend_summary['long_term_trend']}")
        print(f"Medium-term (1yr) Trend: {trend_summary['medium_term_trend']}")
        print(f"Short-term (3mo) Trend: {trend_summary['short_term_trend']}")
        print(f"Momentum Alignment: {trend_summary['momentum_confluence']}")

        # Compare candlestick movements
        self._compare_candlestick_movements(ten_year_data, one_year_data, three_month_data)

        # Compare indicator trends
        self._compare_indicator_trends(ten_year_data, one_year_data, three_month_data)

        return analysis

    def _compare_candlestick_movements(self, ten_year_data, one_year_data, three_month_data):
        """Compare candlestick movements in the last 1 year and 3 months to historical movements in the last 10 years."""
        print("\n[MARKET DATA] Analyzing candlestick pattern significance...")

        # Analyze 3-month movements vs 10-year history
        three_month_movements = self._calculate_candlestick_movements(three_month_data)
        ten_year_movements = self._calculate_candlestick_movements(ten_year_data)

        # Find and classify matches
        three_month_matches = self._find_matching_movements(three_month_movements, ten_year_movements)
        bull_matches_3m = sum(1 for m in three_month_matches if m == 'Bullish')
        bear_matches_3m = len(three_month_matches) - bull_matches_3m
    
        print(f"[MARKET DATA] 3-month patterns vs 10yr history: "
            f"{colored(f'{bull_matches_3m} bullish', 'green')} | "
            f"{colored(f'{bear_matches_3m} bearish', 'red')} | "
            f"Dominant bias: {colored('BULLISH' if bull_matches_3m > bear_matches_3m else 'BEARISH', 'yellow' if bull_matches_3m == bear_matches_3m else 'green' if bull_matches_3m > bear_matches_3m else 'red')}")

        # Analyze 1-year movements vs 10-year history
        one_year_movements = self._calculate_candlestick_movements(one_year_data)
        one_year_matches = self._find_matching_movements(one_year_movements, ten_year_movements)
        bull_matches_1y = sum(1 for m in one_year_matches if m == 'Bullish')
        bear_matches_1y = len(one_year_matches) - bull_matches_1y
    
        print(f"[MARKET DATA] 1-year patterns vs 10yr history: "
            f"{colored(f'{bull_matches_1y} bullish', 'green')} | "
            f"{colored(f'{bear_matches_1y} bearish', 'red')} | "
            f"Dominant bias: {colored('BULLISH' if bull_matches_1y > bear_matches_1y else 'BEARISH', 'yellow' if bull_matches_1y == bear_matches_1y else 'green' if bull_matches_1y > bear_matches_1y else 'red')}")

        # Add comparative analysis
        if bull_matches_3m > bull_matches_1y:
            print(self.format_message("[TREND ANALYSIS] Recent 3-month bullish patterns exceed 1-year trend - potential strengthening momentum"))
        elif bear_matches_3m > bear_matches_1y:
            print(self.format_message("[TREND ANALYSIS] Recent 3-month bearish patterns exceed 1-year trend - potential weakening momentum"))

    def _calculate_candlestick_movements(self, data):
        """Calculate candlestick movements (e.g., bullish/bearish trends) based on OHLC data."""
        movements = []
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['open'].iloc[i]:
                movements.append("Bullish")
            elif data['close'].iloc[i] < data['open'].iloc[i]:
                movements.append("Bearish")
            else:
                movements.append("Neutral")
        return movements

    def _find_matching_movements(self, recent_movements, historical_movements):
        """Find matching candlestick movements between recent and historical data."""
        matches = []
        for i in range(len(recent_movements)):
            if recent_movements[i] == historical_movements[i % len(historical_movements)]:
                matches.append(recent_movements[i])
        return matches

    def _compare_indicator_trends(self, ten_year_data, one_year_data, three_month_data):
        """Compare indicator trends (e.g., RSI, MACD) in the last 1 year and 3 months to historical trends in the last 10 years."""
        print("\n[MARKET DATA] Comparing indicator trends...")

        # Calculate indicator trends for each period
        ten_year_indicators = self._calculate_indicator_trends(ten_year_data)
        one_year_indicators = self._calculate_indicator_trends(one_year_data)
        three_month_indicators = self._calculate_indicator_trends(three_month_data)

        # Compare 3-month indicator trends to 10-year history
        three_month_matches = self._find_matching_indicator_trends(three_month_indicators, ten_year_indicators)
        print(f"[MARKET DATA] 3-month indicator trends matching 10-year history: {len(three_month_matches)}")

        # Compare 1-year indicator trends to 10-year history
        one_year_matches = self._find_matching_indicator_trends(one_year_indicators, ten_year_indicators)
        print(f"[MARKET DATA] 1-year indicator trends matching 10-year history: {len(one_year_matches)}")

    def _calculate_indicator_trends(self, data):
        """Calculate indicator trends (e.g., RSI, MACD) for a given dataset."""
        trends = {
            'rsi': talib.RSI(data['close'], timeperiod=14).iloc[-1],
            'macd': talib.MACD(data['close'])[0].iloc[-1],
            'sma50_vs_sma200': (data['close'].rolling(50).mean().iloc[-1] / data['close'].rolling(200).mean().iloc[-1] - 1) * 100
        }
        return trends

    def _find_matching_indicator_trends(self, recent_trends, historical_trends):
        """Find matching indicator trends between recent and historical data."""
        matches = []
        for key in recent_trends:
            if recent_trends[key] > historical_trends[key]:
                matches.append(f"Bullish {key}")
            elif recent_trends[key] < historical_trends[key]:
                matches.append(f"Bearish {key}")
            else:
                matches.append(f"Neutral {key}")
        return matches
  
    def calculate_indicators(self, data):
        """Compute key indicators for options trading on provided data."""
        data['SMA20'] = talib.SMA(data['close'], timeperiod=20)
        data['SMA50'] = talib.SMA(data['close'], timeperiod=50)
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        macd, macd_signal, _ = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        upper, middle, lower = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        data['UpperBB'] = upper
        data['LowerBB'] = lower
        return data

    def start_trading(self):
        """Main function to run the trading bot."""
        while True:
            self.data_daily = self.fetch_data("daily")
            self.data_hourly = self.fetch_data("hourly")

            if self.data_hourly is not None:
                try:
                    fib_signals = self.fib_strategy_1hour(self.data_hourly)
                    for signal in fib_signals:
                        print(self.format_message(signal))
                except Exception as e:
                    print(f"[ERROR] Failed to run 1 hour Fib strategy: {str(e)}")
                    
            if self.data_daily is not None:
                trend = self.determine_market_trend(self.data_daily)
                print(f"[MARKET DATA] Market Trend: {trend}")

                # Fundamental Analysis
                fundamental_data = self.fetch_fundamental_data()
                if fundamental_data:
                    self.log_fundamentals(fundamental_data)

                # Run Fibonacci and Bollinger Bands analysis
                self.trade_idea(self.data_daily)
            else:
                print("[ERROR] Failed to fetch daily data. Skipping analysis.")

            # Ask user for another ticker or exit
            another_ticker = input("Do you want to analyze another ticker? (yes/no): ").strip().lower()
            if another_ticker == "yes":
                self.ticker = input("Enter the new ticker symbol: ")
                # Fetch new data immediately for the new ticker
                print("[LOG] Fetching data for the new ticker...")
                self.data_daily = self.fetch_data("daily")
                if self.data_daily is not None:
                    trend = self.determine_market_trend(self.data_daily)
                    print(f"[LOG] Market Trend: {trend}")
                    self.trade_idea(self.data_daily)
                else:
                    print("[ERROR] Failed to fetch data for the new ticker.")
            else:
                print("[LOG] Exiting...")
                break

# Modified example usage at bottom of file:
if __name__ == "__main__":
    tickers_input = input("Enter the ticker symbol(s) (comma separated for multiple): ")
    tickers = [t.strip() for t in tickers_input.split(",")]
    
    # Initialize the TradingBot class
    bot = TradingBot()
    
    # Validate the API key
    bot.validate_key()
    
    for t in tickers:
        print(f"\n======== Ticker: {t} ========")
        bot.ticker = t
        data = bot.fetch_data("daily")
        if data is None:
            print(f"[ERROR] Could not fetch data for ticker {t}.")
            continue
        trend = bot.determine_market_trend(data)
        print(f"[LOG] Market Trend: {trend}")
        bot.trade_idea(data)
