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
        """Format 'bullish', 'bearish', 'call', 'calls', 'put', 'puts' with appropriate colors."""
        # Highlight 'bullish' in green
        message = re.sub(
            r'(bullish)',
            lambda m: colored(m.group(1), 'white', 'on_green', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
        # Highlight 'bearish' in red
        message = re.sub(
            r'(bearish)',
            lambda m: colored(m.group(1), 'white', 'on_red', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
        # Highlight 'call' and 'calls' in green
        message = re.sub(
            r'\b(calls?)\b',
            lambda m: colored(m.group(1), 'white', 'on_green', attrs=['bold']),
            message,
            flags=re.IGNORECASE
        )
        # Highlight 'put' and 'puts' in red
        message = re.sub(
            r'\b(puts?)\b',
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
                print(self.format_message("[TRADE IDEA] Regular bearish divergence detected. Potential reversal down."))
            elif (latest_close < prev_close and latest_k > prev_k):
                print(self.format_message("[TRADE IDEA] Regular bullish divergence detected. Potential reversal up."))
            
            # Hidden divergence (continuation)
            if (latest_close > prev_close and latest_k > prev_k):
                print(self.format_message("[TRADE IDEA] Hidden bullish divergence detected. Potential continuation up."))
            elif (latest_close < prev_close and latest_k < prev_k):
                print(self.format_message("[TRADE IDEA] Hidden bearish divergence detected. Potential continuation down."))

        # Determine trade decision
        if latest_k < 20 and latest_d < 20:
            print(self.format_message("[TRADE IDEA] The stochastic Oscillator suggests oversold conditions, bullish signal."))
            return "CALL"
        elif latest_k > 80 and latest_d > 80:
            print(self.format_message("[TRADE IDEA] The stochastic Oscillator suggests overbought conditions, bearish signal."))
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
            print(self.format_message("[TRADE IDEA] 14-period RSI indicates overbought conditions, bearish signal."))
        elif latest_rsi_14 < 30:
            print(self.format_message("[TRADE IDEA] 14-period RSI indicates oversold conditions, bullish signal."))
        else:
            print("[TRADE IDEA] 14-period RSI is neutral, no strong signal.")
        
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
    
    def trade_idea(self, data):
        """Determine a trade idea based on Fibonacci retracement, Bollinger Bands, RSI, MACD, and ATR."""
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

        print(f"[LOG] ADX: {adx_dmi['ADX']:.2f}, +DI: {adx_dmi['Plus_DI']:.2f}, -DI: {adx_dmi['Minus_DI']:.2f}")

    # Calculate VWAP and log
        vwap = self.calculate_vwap(data)
        current_vwap = vwap.iloc[-1]
        print(f"[LOG] Current VWAP: {current_vwap:.2f}")
    
    # Calculate ATR for stop loss and take profit levels
        atr_levels = self.calculate_atr(data)
        atr = atr_levels["ATR"]
        stop_loss_long = atr_levels["stop_loss_long"]
        take_profit_long = atr_levels["take_profit_long"]
        stop_loss_short = atr_levels["stop_loss_short"]
        take_profit_short = atr_levels["take_profit_short"]

        print(f"[LOG] Current ATR: {atr}")

        print("[LOG] Fibonacci Levels:")
        for level, price in yearly_fib_levels.items():
            print(f"{level}: {price}")
        print("[LOG] Fibonacci Levels:")
        for level, price in fib_levels_90days.items():
            print(f"{level}: {price}")

        print("[LOG] Bollinger Bands:")
        print(f"Upper Band: {upper_band.iloc[-1]}")
        print(f"Lower Band: {lower_band.iloc[-1]}")

        print("[LOG] Support and Resistance Levels:")
        for period, levels in support_resistance_levels.items():
            print(f"{period} day Support Level: {levels[0]}")
            print(f"{period} day Resistance Level: {levels[1]}")

        # Add Mag7 Moving Average analysis
        mag7_analysis = self.calculate_mag7_moving_averages(data)
        if mag7_analysis:
            print(f"[LOG] SMA20: {mag7_analysis['SMA20']:.2f}")
            print(f"[LOG] SMA50: {mag7_analysis['SMA50']:.2f}")
            sma20 = mag7_analysis['SMA20']
            sma50 = mag7_analysis['SMA50']
        else:
            sma20 = None
            sma50 = None

        # Calculate additional RSI periods
        data['RSI_7'] = talib.RSI(data['close'], timeperiod=7)
        data['RSI_14'] = talib.RSI(data['close'], timeperiod=14)
        data['RSI_21'] = talib.RSI(data['close'], timeperiod=21)
        
        latest_rsi_7 = data['RSI_7'].iloc[-1]
        latest_rsi_21 = data['RSI_21'].iloc[-1]
        
        # Log all RSI values clearly
        print("\n[LOG] RSI Levels:")
        print(f"7-period RSI: {data['RSI_7'].iloc[-1]:.2f}")
        print(f"14-period RSI: {data['RSI_14'].iloc[-1]:.2f}")
        print(f"21-period RSI (Market Trend): {data['RSI'].iloc[-1]:.2f}")

    # Collect trade ideas
        trade_ideas = []
        current_price = data['close'].iloc[-1]
        print(f"[LOG] Current Price: {current_price}")

        # ADDED: Recent price and volume logging
        recent_high = data['high'].tail(30).max()  # 30-day high
        recent_low = data['low'].tail(30).min()    # 30-day low
        current_volume = data['volume'].iloc[-1]
        
        print(f"\n[LOG] Recent Price & Volume Analysis:")
        print(f"30-Day High: {recent_high:.2f}")
        print(f"30-Day Low: {recent_low:.2f}")
        print(f"Current Volume: {current_volume:,}")  # Format with commas

        # Calculate and log volatility (ADDED SECTION)
        daily_returns = data['close'].pct_change().dropna()
        if len(daily_returns) >= 30:
            historical_volatility = daily_returns.tail(30).std()
            annualized_volatility = historical_volatility * np.sqrt(252)  # 252 trading days/year
            print(f"[LOG] 30-Day Historical Volatility (Daily): {historical_volatility:.4f}")
            print(f"[LOG] Annualized Volatility: {annualized_volatility:.4f}")
        else:
            print("[LOG] Not enough data to calculate 30-day historical volatility")

        if rvi > signal:
            print("[TRADE IDEA] RVI indicates bullish momentum. Consider long positions.")
            trade_ideas.append("[TRADE IDEA] RVI crossover suggests buying opportunity.")
        elif rvi < signal:
            print("[TRADE IDEA] RVI indicates bearish momentum. Consider short positions.")
            trade_ideas.append("[TRADE IDEA] RVI crossover suggests selling opportunity.")

    # Log Coppock Curve trade idea
        if coppock > 0:
            print("[TRADE IDEA] Coppock Curve turning positive. Possible long-term bullish trend forming.")
            trade_ideas.append("[TRADE IDEA] Coppock Curve suggests long-term buying opportunity.")
        else:
            print("[TRADE IDEA] Coppock Curve remains negative. No strong bullish trend yet.")

    # Log DPO trade idea
        if dpo > 0:
            print("[TRADE IDEA] DPO is positive. Short-term bullish cycle detected.")
            trade_ideas.append("[TRADE IDEA] DPO suggests a buying opportunity for short-term swing trades.")
        elif dpo < 0:
            print("[TRADE IDEA] DPO is negative. Short-term bearish cycle detected.")
            trade_ideas.append("[TRADE IDEA] DPO suggests a selling opportunity for short-term swing trades.")

    # Log KVO trade idea
        if kvo > 0:
            print("[TRADE IDEA] KVO indicates positive money flow. Potential buying pressure detected.")
            trade_ideas.append("[TRADE IDEA] KVO suggests institutional buying, consider long positions.")
        elif kvo < 0:
            print("[TRADE IDEA] KVO indicates negative money flow. Potential selling pressure detected.")
            trade_ideas.append("[TRADE IDEA] KVO suggests institutional selling, consider short positions.")

        print(f"[TRADE IDEA] ADX and DMI Trend Analysis: {adx_dmi['Trend Signal']}")
        if adx_dmi['Trend Signal'] == "Strong Bullish Trend":
            trade_ideas.append("[TRADE IDEA] Consider long positions as ADX confirms a strong bullish trend.")
        elif adx_dmi['Trend Signal'] == "Strong Bearish Trend":
            trade_ideas.append("[TRADE IDEA] Consider short positions as ADX confirms a strong bearish trend.")

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

    # VWAP-based signals
        current_price = data['close'].iloc[-1]
    
    # Bullish conditions
        if current_price > current_vwap:
            trade_ideas.append("[TRADE IDEA] Price above VWAP → Look for bullish pullbacks.")
        # Check if VWAP is acting as support
            if data['low'].iloc[-1] <= current_vwap:
                trade_ideas.append("[TRADE IDEA] VWAP acting as support. Confirm with RSI/MACD.")
        # Check SMA alignment for Mag7 stocks
            if sma20 is not None and sma50 is not None:
                if current_price > sma20 and current_price > sma50:
                    trade_ideas.append("[TRADE IDEA] Bullish trend confirmed: Price above SMA20 & SMA50.")
    
    # Bearish conditions
        elif current_price < current_vwap:
            trade_ideas.append("[TRADE IDEA] Price below VWAP → Look for bearish rejections.")
        # Check if VWAP is acting as resistance
            if data['high'].iloc[-1] >= current_vwap:
                trade_ideas.append("[TRADE IDEA] VWAP acting as resistance. Confirm with RSI/MACD.")
        # Check SMA alignment for Mag7 stocks
            if sma20 is not None and sma50 is not None:
                if current_price < sma20 and current_price < sma50:
                    trade_ideas.append("[TRADE IDEA] Bearish trend confirmed: Price below SMA20 & SMA50.")

    # Bollinger Bands conditions
        if current_price > upper_band.iloc[-1]:
            trade_ideas.append("[TRADE IDEA] Bearish, price above upper Bollinger Band.")
        elif current_price < lower_band.iloc[-1]:
            trade_ideas.append("[TRADE IDEA] Bullish, price below lower Bollinger Band.")
        else:
            print("[TRADE IDEA] Price is within Bollinger Bands. No strong signal.")

        if mag7_analysis:
            trade_ideas.append(
                f"[TRADE IDEA] Mag7 Moving Averages: {mag7_analysis['signal']} - {mag7_analysis['reason']}"
            )
    # Fibonacci retracement conditions
        if current_price > yearly_fib_levels['50.0%']:
            trade_ideas.append("[TRADE IDEA] Bullish, price above 50% yearly Fibonacci retracement.")
        if yearly_fib_levels['38.2%'] < current_price <= yearly_fib_levels['50.0%']:
            trade_ideas.append("[TRADE IDEA] Slightly bullish, price near 38.2% yearly Fibonacci retracement.")
        if yearly_fib_levels['23.6%'] < current_price <= yearly_fib_levels['38.2%']:
            trade_ideas.append("[TRADE IDEA] Slightly bearish, price near 23.6% yearly Fibonacci retracement.")
        if current_price <= yearly_fib_levels['23.6%']:
            trade_ideas.append("[TRADE IDEA] Bearish, price below 23.6% yearly Fibonacci retracement.")

        if current_price > fib_levels_90days['50.0%']:
            trade_ideas.append("[TRADE IDEA] Bullish, price above 50% 90 days Fibonacci retracement.")
        if fib_levels_90days['38.2%'] < current_price <= fib_levels_90days['50.0%']:
            trade_ideas.append("[TRADE IDEA] Slightly bullish, price near 38.2% 90 days Fibonacci retracement.")
        if fib_levels_90days['23.6%'] < current_price <= fib_levels_90days['38.2%']:
            trade_ideas.append("[TRADE IDEA] Slightly bearish, price near 23.6% 90 days Fibonacci retracement.")
        if current_price <= fib_levels_90days['23.6%']:
            trade_ideas.append("[TRADE IDEA] Bearish, price below 23.6% 90 days Fibonacci retracement.")

    # RSI-based trade decision
        rsi_signal = self.rsi_trade_decision(data)
        if rsi_signal == "Bullish":
            trade_ideas.append("[TRADE IDEA] 14-period RSI indicates oversold conditions, suggesting a bullish opportunity.")
        elif rsi_signal == "Bearish":
            trade_ideas.append("[TRADE IDEA] 14-period RSI indicates overbought conditions, suggesting a bearish opportunity.")

        # RSI 7 trade idea log
        if latest_rsi_7 > 70:
            print("[TRADE IDEA] 7-period RSI indicates overbought conditions, bearish.")
        elif latest_rsi_7 < 30:
            print("[TRADE IDEA] 7-period RSI indicates oversold conditions, bullish.")
        else:
            print("[TRADE IDEA] 7-period RSI is neutral, no strong signal.")
        
        # RSI 21 trade idea log
        if latest_rsi_21 > 70:
            print("[TRADE IDEA] 21-period RSI indicates overbought conditions, bearish.")
        elif latest_rsi_21 < 30:
            print("[TRADE IDEA] 21-period RSI indicates oversold conditions, bullish.")
        else:
            print("[TRADE IDEA] 21-period RSI is neutral, no strong signal.")
        
        # RSI convergence logic
        if latest_rsi_7 > latest_rsi_21:
            print("[TRADE IDEA] 7-period RSI above 21-period RSI: Bullish momentum detected.")
        elif latest_rsi_7 < latest_rsi_21:
            print("[TRADE IDEA] 7-period RSI below 21-period RSI: Bearish momentum detected.")
        else:
            print("[TRADE IDEA] RSI 7 and 21 are equal: Neutral trend.")

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


    # ATR-based trade ideas for calls and puts
        trade_ideas.append(f"[TRADE IDEA] For LONG CALL positions, consider a stop loss at {stop_loss_long:.2f} and take profit at {take_profit_long:.2f}.")
        trade_ideas.append(f"[TRADE IDEA] For LONG PUT positions, consider a stop loss at {stop_loss_short:.2f} and take profit at {take_profit_short:.2f}.")
        trade_ideas.append(f"[TRADE IDEA] For SHORT PUT positions, consider a stop loss at {stop_loss_short:.2f} and take profit at {take_profit_short:.2f}.")
        trade_ideas.append(f"[TRADE IDEA] For SHORT CALL positions, consider a stop loss at {stop_loss_long:.2f} and take profit at {take_profit_long:.2f}.")

    # Log all trade ideas
        if trade_ideas:
            for idea in trade_ideas:
                print(self.format_message(idea))
        else:
            print(self.format_message("[TRADE IDEA] No strong trade ideas generated based on the current analysis."))

        return trade_ideas
     
    def start_trading(self):
        """Main function to run the trading bot."""
        while True:
            print("[LOG] Fetching daily data...")
            self.data_daily = self.fetch_data("daily")

            if self.data_daily is not None:
                trend = self.determine_market_trend(self.data_daily)
                print(f"[LOG] Market Trend: {trend}")

            # Run Fibonacci and Bollinger Bands analysis
            self.trade_idea(self.data_daily)

        # Interactive analysis after processing the provided tickers
        while True:
            another = input("Do you want to analyze another ticker? (yes/no): ").strip().lower()
            if another != "yes":
                break
            new_ticker = input("Enter the new ticker symbol: ")
            print(f"\n======== Ticker: {new_ticker} ========")
            self.ticker = new_ticker
            data = self.fetch_data("daily")
            if data is None:
                print(f"[ERROR] Could not fetch data for ticker {new_ticker}.")
                continue
            trend = self.determine_market_trend(data)
            print(f"[LOG] Market Trend: {trend}")
            self.trade_idea(data)

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