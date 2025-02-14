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

# Configure rich console
console = Console()

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
trade_buffer = deque(maxlen=100)
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

# Connect to trade server
try:
    sio.connect('http://localhost:8000', transports=['websocket'])
except Exception as e:
    print(f"Could not connect to trade server: {e}")
    with log_lock:
        log_buffer.append(LogRecord(
            name="socketio",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=f"Failed to connect to trade server: {e}",
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
    """Generate trade log panel with added debug output"""
    text = Text()
    with trade_lock:
        # Debug: Show the number of trades received
        debug_line = f"DEBUG: trade_buffer length: {len(trade_buffer)}\n"
        text.append(debug_line, style="bold red")
        
        if not trade_buffer:
            text.append("No trades received yet.\n", style="italic yellow")
        
        # List each trade in the buffer
        for trade in trade_buffer:
            time_str = trade['timestamp'].strftime("%H:%M:%S")
            text.append(f"[{time_str}] ", style=STYLES["timestamp"])
            text.append(trade['message'] + "\n", style=STYLES["trade"])
    
    return Panel(text, title="Trade Log", style="bright_white")

def main():
    """Main execution with rich interface"""
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