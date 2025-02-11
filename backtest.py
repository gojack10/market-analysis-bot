import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta

def fetch_data_from_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
    data.sort_index(inplace=True)  # Ensure data is sorted oldest-first
    return data

def calculate_indicators(data):
    closes = data['4. close']
    highs = data['2. high']
    lows = data['3. low']
    volumes = data['5. volume']

    data['RSI'] = ta.RSI(closes)
    macd, macd_signal, _ = ta.MACD(closes)
    data['MACD'] = macd
    data['MACD_Signal'] = macd_signal
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'] = ta.BBANDS(closes)
    for period in [5, 10, 20]:
        data[f'EMA_{period}'] = ta.EMA(closes, timeperiod=period)
    data['Stoch_Slowk'], data['Stoch_Slowd'] = ta.STOCH(highs, lows, closes)
    data['ATR'] = ta.ATR(highs, lows, closes, timeperiod=14)
    data['TRIX'] = ta.TRIX(closes, timeperiod=15)
    data['Volume_MA20'] = ta.SMA(volumes, timeperiod=20)
    data['EMA_8'] = ta.EMA(closes, timeperiod=8)
    data['VWAP'] = (volumes * (highs + lows + closes) / 3).cumsum() / volumes.cumsum()
    data['OBV'] = ta.OBV(closes, volumes)
    return data.dropna()

def calculate_fibonacci_levels(data, period=14):
    high = data['2. high'].rolling(window=period).max()
    low = data['3. low'].rolling(window=period).min()
    diff = high - low
    data['Fib_38_2'] = high - (diff * 0.382)
    data['Fib_61_8'] = high - (diff * 0.618)
    return data
    
class StrategyEngine:
    def generate_signals_strategy1(self, df):
        df['Signal_RSI'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] > 65, -1, 0))
        df['Signal_MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        df['Signal_Fib'] = np.where((df['4. close'] < df['Fib_38_2']) & (df['4. close'] > df['Fib_61_8']), 1, 0)
        df['Buy_Strategy1'] = (df['Signal_RSI'] == 1) & (df['Signal_MACD'] == 1) & (df['Signal_Fib'] == 1)
        df['Sell_Strategy1'] = (df['Signal_RSI'] == -1) & (df['Signal_MACD'] == -1) & (df['Signal_Fib'] == 1)
        return df
    
    def generate_signals_strategy2(self, df):
        df['Signal_RSI'] = np.where(df['RSI'] < 35, 1, np.where(df['RSI'] > 65, -1, 0))
        df['Signal_MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        df['Buy_Strategy2'] = (df['Signal_RSI'] == 1) & (df['Signal_MACD'] == 1)
        df['Sell_Strategy2'] = (df['Signal_RSI'] == -1) & (df['Signal_MACD'] == -1)
        return df
    
    def generate_signals_strategy3(self, df):
        df['Signal_VWAP'] = np.where(df['4. close'] > df['VWAP'], 1, -1)
        df['Signal_OBV'] = np.where(df['OBV'].diff() > 0, 1, -1)
        df['Signal_Fib'] = np.where((df['4. close'] < df['Fib_38_2']) & (df['4. close'] > df['Fib_61_8']), 1, 0)
        df['Buy_Strategy3'] = (df['Signal_VWAP'] == 1) & (df['Signal_OBV'] == 1) & (df['Signal_Fib'] == 1)
        df['Sell_Strategy3'] = (df['Signal_VWAP'] == -1) & (df['Signal_OBV'] == -1) & (df['Signal_Fib'] == 1)
        return df

def backtest(df, buy_column, sell_column, long_sl_multiplier, long_tp_multiplier, short_sl_multiplier, short_tp_multiplier, sl_return_long, tp_return_long, sl_return_short, tp_return_short, unresolved_return):
    trades = []
    active_trades = []  # (entry_price, sl, tp, trade_type)

    for i in range(len(df)):
        new_active_trades = []
        for trade in active_trades:
            entry_price, sl, tp, trade_type = trade
            current_low = df['3. low'].iloc[i]
            current_high = df['2. high'].iloc[i]

            if trade_type == 'long':
                if current_low <= sl:
                    trades.append(sl_return_long)
                elif current_high >= tp:
                    trades.append(tp_return_long)
                else:
                    new_active_trades.append(trade)
            elif trade_type == 'short':
                if current_high >= sl:
                    trades.append(sl_return_short)
                elif current_low <= tp:
                    trades.append(tp_return_short)
                else:
                    new_active_trades.append(trade)
        active_trades = new_active_trades

        if df[buy_column].iloc[i]:
            entry_price = df['4. close'].iloc[i]
            sl = entry_price * long_sl_multiplier
            tp = entry_price * long_tp_multiplier
            active_trades.append((entry_price, sl, tp, 'long'))
        if df[sell_column].iloc[i]:
            entry_price = df['4. close'].iloc[i]
            sl = entry_price * short_sl_multiplier
            tp = entry_price * short_tp_multiplier
            active_trades.append((entry_price, sl, tp, 'short'))

    for trade in active_trades:
        entry_price, sl, tp, trade_type = trade
        exit_price = df['4. close'].iloc[-1]
        if trade_type == 'long':
            if exit_price >= tp:
                trades.append(tp_return_long)
            elif exit_price <= sl:
                trades.append(sl_return_long)
            else:
                trades.append(unresolved_return)
        elif trade_type == 'short':
            if exit_price <= tp:
                trades.append(tp_return_short)
            elif exit_price >= sl:
                trades.append(sl_return_short)
            else:
                trades.append(unresolved_return)

    return trades

# Execute Strategy
file_path = 'AAPL.csv'
data = fetch_data_from_csv(file_path)
data = calculate_indicators(data)
data = calculate_fibonacci_levels(data)
data = data.dropna()

engine = StrategyEngine()
data = engine.generate_signals_strategy1(data)
data = engine.generate_signals_strategy2(data)
data = engine.generate_signals_strategy3(data)

strategy_params = {
    'Strategy1': {
        'buy_column': 'Buy_Strategy1',
        'sell_column': 'Sell_Strategy1',
        'long_sl_multiplier': 0.70,
        'long_tp_multiplier': 1.01,
        'short_sl_multiplier': 1.30,  # Keeping it at 30% loss
        'short_tp_multiplier': 0.99,
        'sl_return_long': -0.30,
        'tp_return_long': 0.435,
        'sl_return_short': -0.30,
        'tp_return_short': 0.01,
        'unresolved_return': -0.30
    },
    'Strategy2': {
        'buy_column': 'Buy_Strategy2',
        'sell_column': 'Sell_Strategy2',
        'long_sl_multiplier': 0.90,
        'long_tp_multiplier': 1.01,
        'short_sl_multiplier': 1.30,  # Changed to 30% loss
        'short_tp_multiplier': 0.99,
        'sl_return_long': -0.10,
        'tp_return_long': 1.00,
        'sl_return_short': -0.30,  # Updated to reflect the larger stop-loss
        'tp_return_short': 0.01,
        'unresolved_return': -0.30
    },
    'Strategy3': {
        'buy_column': 'Buy_Strategy3',
        'sell_column': 'Sell_Strategy3',
        'long_sl_multiplier': 0.90,
        'long_tp_multiplier': 1.01,
        'short_sl_multiplier': 1.30,  # Changed to 30% loss
        'short_tp_multiplier': 0.99,
        'sl_return_long': -0.10,
        'tp_return_long': 1.00,
        'sl_return_short': -0.30,  # Updated to reflect the larger stop-loss
        'tp_return_short': 0.01,
        'unresolved_return': -0.30
    }
}

results = {}
for strategy_name, params in strategy_params.items():
    trades = backtest(data, **params)
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    results[strategy_name] = {
        'Total Trades': len(trades),
        'Win Rate': f"{win_rate:.1f}%",
        'Average Return': f"{np.mean(trades)*100:.1f}%" if trades else "N/A"
    }

for strategy, metrics in results.items():
    print(f"\n{strategy} Results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")