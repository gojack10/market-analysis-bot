# trade_gui.py

import upgrade3
from upgrade3 import *
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.style import Style
import logging
from logging import Handler, LogRecord
from collections import deque
from threading import Thread, Lock
from datetime import datetime
import socketio
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure rich console
console = Console()

# Socket.IO server URL from environment variable
TRADE_SERVER_URL = os.getenv('TRADE_SERVER_URL', 'http://localhost:8000')

# Custom color styles
STYLES = {
    "call": Style(color="bright_green", bold=True),
    "put": Style(color="bright_red", bold=True),
    "neutral": Style(color="yellow"),
    "info": Style(color="cyan"),
    "warning": Style(color="yellow"),
    "error": Style(color="red"),
    "timestamp": Style(color="magenta"),
    "symbol": Style(color="bright_cyan"),
    "trade": Style(color="bright_white", bold=True)
}

# Log buffers for terminal display
log_buffer = deque(maxlen=100)
# Enforce rotation after 4 trades, so the 5th pushes out the oldest
trade_buffer = deque(maxlen=4)
log_lock = Lock()
trade_lock = Lock()

# Initialize Socket.IO client
sio = socketio.Client()

@sio.on('connect')
def on_connect():
    """Handle successful connection"""
    print("\n=== Socket.IO Connected ===")
    print(f"Connected to trade server at {datetime.now().strftime('%H:%M:%S')}")
    print("============================")
    with log_lock:
        log_buffer.append(LogRecord(
            name="socketio",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Connected to trade server",
            args=(),
            exc_info=None,
            created=time.time()
        ))

@sio.on('disconnect')
def on_disconnect():
    """Handle disconnection"""
    print("\n=== Socket.IO Disconnected ===")
    print(f"Disconnected at {datetime.now().strftime('%H:%M:%S')}")
    print("==============================")
    with log_lock:
        log_buffer.append(LogRecord(
            name="socketio",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Disconnected from trade server",
            args=(),
            exc_info=None,
            created=time.time()
        ))

@sio.on('new_trade')
def on_new_trade(message):
    """Handle incoming trade messages"""
    print("\n=== Socket.IO Event Received ===")
    print(f"Received trade at {datetime.now().strftime('%H:%M:%S')}:")
    print(f"Message content: {message}")
    print("================================")
    
    with trade_lock:
        trade_buffer.append({
            'timestamp': datetime.now(),
            'message': message
        })

@sio.on('close_trade')
def on_close_trade(message):
    """Handle trade close messages"""
    print("\n=== Socket.IO Close Trade Event ===")
    print(f"Received close trade at {datetime.now().strftime('%H:%M:%S')}:")
    print(f"Message content: {message}")
    print("==================================")
    
    with trade_lock:
        trade_buffer.append({
            'timestamp': datetime.now(),
            'message': message
        })

@sio.on('debug_log')
def on_debug_log(message):
    """Handle debug log messages"""
    print("\n=== Socket.IO Debug Log Event ===")
    print(f"Received debug log at {datetime.now().strftime('%H:%M:%S')}:")
    print(f"Message content: {message}")
    print("=================================")
    
    with trade_lock:
        # Look for the trigger text anywhere in the message.
        if "Trade list memory cleared" in message:
            trade_buffer.clear()
        trade_buffer.append({
            'timestamp': datetime.now(),
            'message': message
        })

# Connect to trade server
try:
    sio.connect(TRADE_SERVER_URL, transports=['websocket'])
except Exception as e:
    print(f"Could not connect to trade server at {TRADE_SERVER_URL}: {e}")
    with log_lock:
        log_buffer.append(LogRecord(
            name="socketio",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=f"Failed to connect to trade server at {TRADE_SERVER_URL}: {e}",
            args=(),
            exc_info=None,
            created=time.time()
        ))

class RichLogHandler(Handler):
    def emit(self, record):
        with log_lock:
            log_buffer.append(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichLogHandler()]
)

class EnhancedStrategyEngine(upgrade3.RealtimeStrategyEngine):
    """Extended strategy engine with rich logging"""
    
    def generate_signals_strategy1(self, df, symbol):
        df = super().generate_signals_strategy1(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy1', False):
            logging.info(f"[STRAT1] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
        elif latest.get('Sell_Strategy1', False):
            logging.info(f"[STRAT1] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
        else:
            logging.info(f"[STRAT1] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df
    
    def generate_signals_strategy2(self, df, symbol):
        df = super().generate_signals_strategy2(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy2', False):
            logging.info(f"[STRAT2] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
        elif latest.get('Sell_Strategy2', False):
            logging.info(f"[STRAT2] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
        else:
            logging.info(f"[STRAT2] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df
    
    def generate_signals_strategy3(self, df, symbol):
        df = super().generate_signals_strategy3(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy3', False):
            logging.info(f"[STRAT3] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
        elif latest.get('Sell_Strategy3', False):
            logging.info(f"[STRAT3] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
        else:
            logging.info(f"[STRAT3] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df

def create_layout():
    """Create terminal layout structure"""
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="logs", size=10)
    )
    layout["main"].split_row(
        Layout(name="status", ratio=2),
        Layout(name="signals", ratio=3)
    )
    layout["logs"].split_row(
        Layout(name="event_log", ratio=1),
        Layout(name="trade_log", ratio=1)
    )
    return layout

def generate_header() -> Panel:
    """Generate top header panel"""
    title = Text(" ALGO TRADING BOT v3.0 ", style="white on blue bold")
    return Panel(title, style="blue")

def generate_status(symbol_data: dict) -> Panel:
    """Generate status panel with market data"""
    table = Table(show_header=False, padding=(0,1))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bright_white")
    
    for symbol, data in symbol_data.items():
        table.add_row(f"[{symbol}] Last Update", data['timestamp'], end_section=True)
        table.add_row("Price", f"{data['close']:.2f}")
        table.add_row("Volume", f"{data['volume']:,.0f}")
        table.add_row("RSI", f"{data['rsi']:.1f}")
        table.add_row("MACD", f"{data['macd']:.4f}")
        table.add_row("VWAP", f"{data['vwap']:.2f}")
        table.add_row("Bollinger", f"{data['bollinger']}")
        table.add_row("Fib Level", data['fib_level'], end_section=True)
    
    return Panel(table, title="Market Data", subtitle="Realtime Stats", style="bright_white")

def generate_signals(signal_data: list) -> Panel:
    """Generate signals panel with proper log parsing"""
    table = Table(
        title="Active Signals",
        show_header=True,
        header_style="bold magenta",
        expand=True
    )
    
    table.add_column("Time", style="cyan")
    table.add_column("Symbol", style="bright_cyan")
    table.add_column("Strategy", style="yellow")
    table.add_column("Signal", justify="right")
    
    # Show the last 6 signal entries
    for entry in signal_data[-6:]:
        time_str = datetime.fromtimestamp(entry.created).strftime('%H:%M:%S')
        log_message = entry.getMessage()
        
        parts = log_message.strip().split()
        strategy = symbol = signal = "N/A"
        
        if len(parts) >= 4:
            strategy = parts[0][1:-1]
            symbol = parts[1]
            signal = f"{parts[2]} {parts[3]}"
            
            # Determine signal style
            if "NO TRADE" in signal:
                signal_style = STYLES["neutral"]
            else:
                signal_style = Style(color="bright_green", bold=True) if "CALL" in signal else Style(color="bright_red", bold=True)
        
        table.add_row(
            time_str,
            Text(symbol, style="bright_cyan"),
            strategy,
            Text(signal.upper(), style=signal_style)
        )
    
    return Panel(table, style="bright_white")

def generate_logs() -> Panel:
    """Generate scrolling log panel"""
    text = Text()
    with log_lock:
        for record in log_buffer:
            style = STYLES.get(record.levelname.lower(), Style())
            time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            text.append(f"[{time_str}] ", style=STYLES["timestamp"])
            text.append(record.getMessage() + "\n", style=style)
    
    return Panel(text, title="Event Log", style="bright_white")

def generate_trade_log() -> Panel:
    """
    Generate trade log panel with a maximum of 4 trades.
    Each trade prints two lines (timestamp + message) => 8 lines max.
    """
    text = Text()
    with trade_lock:
        if not trade_buffer:
            text.append("No trades received yet.\n", style="italic yellow")
        
        # The deque itself is maxlen=4, so only 4 trades will ever be stored
        for trade in trade_buffer:
            time_str = trade['timestamp'].strftime("%H:%M:%S")
            text.append(f"[{time_str}] ", style=STYLES["timestamp"])
            text.append(trade['message'] + "\n", style=STYLES["trade"])
    
    return Panel(text, title="Trade Log", style="bright_white")

def main():
    """Main execution with rich interface"""
    # Start any background music or theme if needed
    Thread(target=play_theme_loop, daemon=True).start()
    
    layout = create_layout()
    engine = EnhancedStrategyEngine()
    symbol_data = {}
    
    with Live(layout, refresh_per_second=10, screen=True):
        while True:
            layout["header"].update(generate_header())
            
            for symbol in SYMBOLS:
                df = fetch_realtime_data(symbol)
                if df is not None and len(df) > 20:
                    df = df.sort_index()
                    df = calculate_indicators(df)
                    df = calculate_fibonacci_levels(df)
                    df = engine.generate_signals_strategy1(df, symbol)
                    df = engine.generate_signals_strategy2(df, symbol)
                    df = engine.generate_signals_strategy3(df, symbol)
                    
                    latest = df.iloc[-1]
                    symbol_data[symbol] = {
                        'timestamp': latest.name.strftime('%H:%M:%S'),
                        'close': latest['close'],
                        'volume': latest['volume'],
                        'rsi': latest.get('RSI', 0),
                        'macd': latest.get('MACD', 0),
                        'vwap': latest.get('VWAP', 0),
                        'bollinger': f"{latest.get('Bollinger_Lower', 0):.2f}-{latest.get('Bollinger_Upper', 0):.2f}",
                        'fib_level': f"38.2%: {latest.get('Fib_38_2', 0):.2f} | 61.8%: {latest.get('Fib_61_8', 0):.2f}"
                    }
            
            layout["status"].update(generate_status(symbol_data))
            layout["signals"].update(generate_signals(list(log_buffer)))
            layout["event_log"].update(generate_logs())
            layout["trade_log"].update(generate_trade_log())
            
            time.sleep(5)

if __name__ == "__main__":
    try:
        from rich import print
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "rich"])
    
    try:
        main()
    finally:
        if sio.connected:
            sio.disconnect()
import upgrade3
from upgrade3 import *
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.style import Style
import logging
from logging import Handler, LogRecord
from collections import deque
from threading import Thread, Lock
from datetime import datetime
import socketio
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure rich console
console = Console()

# Socket.IO server URL from environment variable
TRADE_SERVER_URL = os.getenv('TRADE_SERVER_URL', 'http://localhost:8000')

# Custom color styles
STYLES = {
    "call": Style(color="bright_green", bold=True),
    "put": Style(color="bright_red", bold=True),
    "neutral": Style(color="yellow"),
    "info": Style(color="cyan"),
    "warning": Style(color="yellow"),
    "error": Style(color="red"),
    "timestamp": Style(color="magenta"),
    "symbol": Style(color="bright_cyan"),
    "trade": Style(color="bright_white", bold=True)
}

# Log buffers for terminal display
log_buffer = deque(maxlen=100)
# Enforce rotation after 4 trades, so the 5th pushes out the oldest
trade_buffer = deque(maxlen=4)
ml_buffer = deque(maxlen=50)
log_lock = Lock()
trade_lock = Lock()
ml_lock = Lock()

# Initialize Socket.IO client
sio = socketio.Client()

@sio.on('connect')
def on_connect():
    """Handle successful connection"""
    print("\n=== Socket.IO Connected ===")
    print(f"Connected to trade server at {datetime.now().strftime('%H:%M:%S')}")
    print("============================")
    with log_lock:
        log_buffer.append(LogRecord(
            name="socketio",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Connected to trade server",
            args=(),
            exc_info=None,
            created=time.time()
        ))

@sio.on('disconnect')
def on_disconnect():
    """Handle disconnection"""
    print("\n=== Socket.IO Disconnected ===")
    print(f"Disconnected at {datetime.now().strftime('%H:%M:%S')}")
    print("==============================")
    with log_lock:
        log_buffer.append(LogRecord(
            name="socketio",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Disconnected from trade server",
            args=(),
            exc_info=None,
            created=time.time()
        ))

@sio.on('new_trade')
def on_new_trade(message):
    """Handle incoming trade messages"""
    print("\n=== Socket.IO Event Received ===")
    print(f"Received trade at {datetime.now().strftime('%H:%M:%S')}:")
    print(f"Message content: {message}")
    print("================================")
    
    with trade_lock:
        trade_buffer.append({
            'timestamp': datetime.now(),
            'message': message
        })

@sio.on('close_trade')
def on_close_trade(message):
    """Handle trade close messages"""
    print("\n=== Socket.IO Close Trade Event ===")
    print(f"Received close trade at {datetime.now().strftime('%H:%M:%S')}:")
    print(f"Message content: {message}")
    print("==================================")
    
    with trade_lock:
        trade_buffer.append({
            'timestamp': datetime.now(),
            'message': message
        })
@sio.on('debug_log')
def on_debug_log(message):
    """Handle debug log messages"""
    print("\n=== Socket.IO Debug Log Event ===")
    print(f"Received debug log at {datetime.now().strftime('%H:%M:%S')}:")
    print(f"Message content: {message}")
    print("=================================")
    
    with trade_lock:
        # Look for the trigger text anywhere in the message.
        if "Trade list memory cleared" in message:
            trade_buffer.clear()
        trade_buffer.append({
            'timestamp': datetime.now(),
            'message': message
        })

# Connect to trade server
try:
    sio.connect(TRADE_SERVER_URL, transports=['websocket'])
except Exception as e:
    print(f"Could not connect to trade server at {TRADE_SERVER_URL}: {e}")
    with log_lock:
        log_buffer.append(LogRecord(
            name="socketio",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=f"Failed to connect to trade server at {TRADE_SERVER_URL}: {e}",
            args=(),
            exc_info=None,
            created=time.time()
        ))

class RichLogHandler(Handler):
    def emit(self, record):
        with log_lock:
            log_buffer.append(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichLogHandler()]
)

class EnhancedStrategyEngine(upgrade3.RealtimeStrategyEngine):
    """Extended strategy engine with rich logging and sound alerts"""

    def generate_signals_strategy1(self, df, symbol):
        df = super().generate_signals_strategy1(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy1', False):
            logging.info(f"[STRAT1] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.CALL_SOUND))
        elif latest.get('Sell_Strategy1', False):
            logging.info(f"[STRAT1] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.PUT_SOUND))
        else:
            logging.info(f"[STRAT1] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df

    def generate_signals_strategy2(self, df, symbol):
        df = super().generate_signals_strategy2(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy2', False):
            logging.info(f"[STRAT2] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.CALL_SOUND))
        elif latest.get('Sell_Strategy2', False):
            logging.info(f"[STRAT2] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.PUT_SOUND))
        else:
            logging.info(f"[STRAT2] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df

    def generate_signals_strategy3(self, df, symbol):
        df = super().generate_signals_strategy3(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy3', False):
            logging.info(f"[STRAT3] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.CALL_SOUND))
        elif latest.get('Sell_Strategy3', False):
            logging.info(f"[STRAT3] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.PUT_SOUND))
        else:
            logging.info(f"[STRAT3] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df

    def generate_signals_strategy4(self, df, symbol):
        df = super().generate_signals_strategy4(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy4', False):
            logging.info(f"[STRAT4] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.CALL_SOUND))
        elif latest.get('Sell_Strategy4', False):
            logging.info(f"[STRAT4] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.PUT_SOUND))
        else:
            logging.info(f"[STRAT4] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df
        
    def generate_signals_strategy5(self, df, symbol):
        df = super().generate_signals_strategy5(df)
        latest = df.iloc[-1]
        if latest.get('Buy_Strategy5', False):
            logging.info(f"[STRAT5] {symbol} BUY CALL - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.CALL_SOUND))
        elif latest.get('Sell_Strategy5', False):
            logging.info(f"[STRAT5] {symbol} BUY PUT - {latest.name.strftime('%H:%M:%S')}")
            pygame.mixer.Channel(1).play(pygame.mixer.Sound(upgrade3.PUT_SOUND))
        else:
            logging.info(f"[STRAT5] {symbol} NO TRADE - {latest.name.strftime('%H:%M:%S')}")
        return df

def create_layout():
    """Create terminal layout structure with ML predictions column"""
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="ml_predictions", size=10),
        Layout(name="logs", size=10),
    )
    layout["main"].split_row(
        Layout(name="status", ratio=2),
        Layout(name="signals", ratio=3),
    )
    layout["logs"].split_row(
        Layout(name="event_log", ratio=1),
        Layout(name="trade_log", ratio=1)
    )
    return layout

def generate_ml_predictions() -> Panel:
    """Generate panel for machine learning trend predictions"""
    table = Table(
        title="ML Trend Analysis",
        show_header=True,
        header_style="bold green",
        expand=True
    )
    table.add_column("Time", style="cyan")
    table.add_column("Symbol", style="bright_cyan")
    table.add_column("Prediction", justify="right")
    
    with ml_lock:
        for entry in list(ml_buffer)[-3:]:
            time_str = entry['timestamp']
            symbol = entry['symbol']
            trend = entry['trend']
            trend_style = STYLES["call"] if trend == "UPWARD" else STYLES["put"]
            
            table.add_row(
                Text(time_str, style=STYLES["timestamp"]),
                Text(symbol, style="bright_cyan"),
                Text(trend, style=trend_style)
            )
    
    return Panel(table, style="bright_white")

def generate_header() -> Panel:
    """Generate top header panel"""
    title = Text(" ALGO TRADING BOT v3.0 ", style="white on blue bold")
    return Panel(title, style="blue")



def generate_status(symbol_data: dict) -> Panel:
    """Generate status panel with market data"""
    table = Table(show_header=False, padding=(0,1))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bright_white")
    
    for symbol, data in symbol_data.items():
        table.add_row(f"[{symbol}] Last Update", data['timestamp'], end_section=True)
        table.add_row("Price", f"{data['close']:.2f}")
        table.add_row("Volume", f"{data['volume']:,.0f}")
        table.add_row("RSI", f"{data['rsi']:.1f}")
        table.add_row("MACD", f"{data['macd']:.4f}")
        table.add_row("VWAP", f"{data['vwap']:.2f}")
        table.add_row("Bollinger", f"{data['bollinger']}")
        table.add_row("Fib Level", data['fib_level'], end_section=True)
        
    return Panel(table, title="Market Data", subtitle="Realtime Stats", style="bright_white")

def generate_signals(signal_data: list) -> Panel:
    """Generate signals panel with proper log parsing"""
    table = Table(
        title="Active Signals",
        show_header=True,
        header_style="bold magenta",
        expand=True
    )
    
    table.add_column("Time", style="cyan")
    table.add_column("Symbol", style="bright_cyan")
    table.add_column("Strategy", style="yellow")
    table.add_column("Signal", justify="right")
    
    for entry in signal_data[-15:]:
        time_str = datetime.fromtimestamp(entry.created).strftime('%H:%M:%S')
        log_message = entry.getMessage()
        
        parts = log_message.strip().split()
        strategy = symbol = signal = "N/A"
        
        if len(parts) >= 4:
            strategy = parts[0][1:-1]
            symbol = parts[1]
            signal = f"{parts[2]} {parts[3]}"
            
            # Determine signal style
            if "NO TRADE" in signal:
                signal_style = STYLES["neutral"]
            else:
                signal_style = Style(color="bright_green", bold=True) if "CALL" in signal else Style(color="bright_red", bold=True)
        
        table.add_row(
            time_str,
            Text(symbol, style="bright_cyan"),
            strategy,
            Text(signal.upper(), style=signal_style)
        )
    
    return Panel(table, style="bright_white")

def generate_logs() -> Panel:
    """Generate scrolling log panel"""
    text = Text()
    with log_lock:
        for record in log_buffer:
            style = STYLES.get(record.levelname.lower(), Style())
            time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            text.append(f"[{time_str}] ", style=STYLES["timestamp"])
            text.append(record.getMessage() + "\n", style=style)
    
    return Panel(text, title="Event Log", style="bright_white")

def generate_trade_log() -> Panel:
    """
    Generate trade log panel with a maximum of 4 trades.
    Each trade prints two lines (timestamp + message) => 8 lines max.
    """
    text = Text()
    with trade_lock:
        if not trade_buffer:
            text.append("No trades received yet.\n", style="italic yellow")
        
        # The deque itself is maxlen=4, so only 4 trades will ever be stored
        for trade in trade_buffer:
            time_str = trade['timestamp'].strftime("%H:%M:%S")
            text.append(f"[{time_str}] ", style=STYLES["timestamp"])
            text.append(trade['message'] + "\n", style=STYLES["trade"])
    
    return Panel(text, title="Trade Log", style="bright_white")

def main():
    """Main execution with rich interface and ML integration"""
    Thread(target=upgrade3.play_theme_loop, daemon=True).start()
    
    layout = create_layout()
    engine = EnhancedStrategyEngine()
    ml_model = upgrade3.MachineLearningModel()
    symbol_data = {}
    
    with Live(layout, refresh_per_second=10, screen=True):
        while True:
            layout["header"].update(generate_header())
            
            for symbol in SYMBOLS:
                df = fetch_realtime_data(symbol)
                if df is not None and len(df) > 20:
                    df = df.sort_index()
                    df = calculate_indicators(df)
                    df = calculate_fibonacci_levels(df)
                    df = engine.generate_signals_strategy1(df, symbol)
                    df = engine.generate_signals_strategy2(df, symbol)
                    df = engine.generate_signals_strategy3(df, symbol)
                    df = engine.generate_signals_strategy4(df, symbol)
                    df = engine.generate_signals_strategy5(df, symbol)
                    
                    latest = df.iloc[-1]
                    symbol_data[symbol] = {
                        'timestamp': latest.name.strftime('%H:%M:%S'),
                        'close': latest['close'],
                        'volume': latest['volume'],
                        'rsi': latest.get('RSI', 0),
                        'macd': latest.get('MACD', 0),
                        'vwap': latest.get('VWAP', 0),
                        'bollinger': f"{latest.get('Bollinger_Lower', 0):.2f}-{latest.get('Bollinger_Upper', 0):.2f}",
                        'fib_level': f"38.2%: {latest.get('Fib_38_2', 0):.2f} | 61.8%: {latest.get('Fib_61_8', 0):.2f}"
                    }

                    # ML Prediction processing
                    csv_file = f"{symbol}1.csv"
                    latest_data = [latest[feature] for feature in ml_model.features]
                    prediction = ml_model.predict_trend(csv_file, latest_data)
                    
                    if prediction is not None:
                        trend = 'UPWARD' if prediction[0] == 1 else 'DOWNWARD'
                        timestamp = latest.name.strftime('%H:%M:%S')
                        with ml_lock:
                            ml_buffer.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'trend': trend
                            })
            
            layout["status"].update(generate_status(symbol_data))
            layout["signals"].update(generate_signals(list(log_buffer)))
            layout["ml_predictions"].update(generate_ml_predictions())
            layout["logs"].update(generate_logs())
            layout["event_log"].update(generate_logs())
            layout["trade_log"].update(generate_trade_log())
            
            time.sleep(5)

if __name__ == "__main__":
    try:
        from rich import print
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "rich"])
    
    main()