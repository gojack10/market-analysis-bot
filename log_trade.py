import socketio
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize SocketIO client
sio = socketio.Client()

# Get server URL from environment variable
TRADE_SERVER_URL = os.getenv('TRADE_SERVER_URL', 'http://localhost:8000')

def log_trade(ticker):
    """
    Logs a new trade for the selected ticker.
    Prompts the user for trade type, expiration date, and strike price.
    Emits the trade event and returns the trade details if successful.
    """
    try:
        # Connect to the server with explicit transport setting
        sio.connect(TRADE_SERVER_URL, transports=['websocket'])
        
        print("\n=== Trade Logger ===")
        print(f"Connecting to: {TRADE_SERVER_URL}")
        # Log the ticker selection after connection
        print(f"Selected Ticker: {ticker}")
        
        trade_type = input("Enter trade type (call/put): ").strip().upper()
        exp_date = input("Enter expiration date (e.g., 2-14): ").strip()
        strike_price = input("Enter strike price: ").strip()
        
        # Validate inputs
        if not all([trade_type, exp_date, strike_price]):
            print("Error: All fields are required!")
            return None
        
        if trade_type not in ['CALL', 'PUT']:
            print("Error: Trade type must be either 'call' or 'put'!")
            return None
            
        trade = {
            'ticker': ticker,
            'username': os.getenv('USERNAME', 'unknown'),
            'tradeType': trade_type,
            'expDate': exp_date,
            'strikePrice': strike_price,
            'status': 'open'
        }
        
        # Emit the new trade event
        sio.emit('new_trade', trade)
        
        print(f"\nTrade logged successfully!")
        print(f"Ticker: {ticker}")
        print(f"Type: {trade_type}")
        print(f"Expiration: {exp_date}")
        print(f"Strike: {strike_price}")
        
        # Delay to ensure event propagation before disconnecting
        time.sleep(1)
        
        return trade
        
    except Exception as e:
        print(f"Error connecting to {TRADE_SERVER_URL}: {e}")
        print("Make sure the trade server is running and the TRADE_SERVER_URL is correct")
        return None
    
    finally:
        if sio.connected:
            sio.disconnect()

def close_trade(trade):
    """
    Closes an open trade.
    Emits a 'close_trade' event with the updated trade information.
    """
    try:
        sio.connect(TRADE_SERVER_URL, transports=['websocket'])
        
        trade['status'] = 'closed'
        sio.emit('close_trade', trade)
        
        print(f"\nTrade closed successfully!")
        print(f"Ticker: {trade['ticker']}")
        print(f"Type: {trade['tradeType']}")
        print(f"Expiration: {trade['expDate']}")
        print(f"Strike: {trade['strikePrice']}")
        
        time.sleep(1)
        
    except Exception as e:
        print(f"Error closing trade: {e}")
    finally:
        if sio.connected:
            sio.disconnect()

def debug_clear_memory(trade_memory):
    """
    Clears the trade memory list and emits a debug log event with a descriptive message.
    """
    trade_memory.clear()
    debug_message = "[DEBUG] Trade list memory cleared. All trade logs removed."
    print(debug_message)
    try:
        sio.connect(TRADE_SERVER_URL, transports=['websocket'])
        sio.emit('debug_log', {'message': debug_message})
        time.sleep(1)
    except Exception as e:
        print(f"Error logging debug event: {e}")
    finally:
        if sio.connected:
            sio.disconnect()

def select_ticker():
    """
    Prompts the user to select a ticker using number 1 or 2.
    Returns the ticker symbol.
    """
    ticker = None
    while ticker is None:
        print("\nSelect Ticker:")
        print("1. NVDA")
        print("2. AAPL")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            ticker = "NVDA"
        elif choice == "2":
            ticker = "AAPL"
        else:
            print("Invalid selection. Please enter 1 or 2.")
    return ticker

def main():
    trade_memory = []  # Memory to store trades in progress
    
    while True:
        if not trade_memory:
            print("\nNo trades in progress.")
            ticker = select_ticker()
            trade = log_trade(ticker)
            if trade:
                trade_memory.append(trade)
        else:
            print("\nCurrent Trades in Memory:")
            for idx, trade in enumerate(trade_memory, start=1):
                print(f"{idx}. Ticker: {trade['ticker']}, Type: {trade['tradeType']}, Exp: {trade['expDate']}, Strike: {trade['strikePrice']}, Status: {trade['status']}")
            
            print("\nOptions:")
            print("1. Open New Trade")
            print("2. Close Trade")
            print("3. [DEBUG] CLEAR TRADE LIST MEMORY")
            option = input("Enter option number: ").strip()
            
            if option == "1":
                ticker = select_ticker()
                trade = log_trade(ticker)
                if trade:
                    trade_memory.append(trade)
            elif option == "2":
                if not trade_memory:
                    print("No open trades to close.")
                elif len(trade_memory) == 1:
                    trade_to_close = trade_memory.pop(0)
                    close_trade(trade_to_close)
                else:
                    try:
                        index = int(input("Enter the trade number to close: ").strip())
                        if 1 <= index <= len(trade_memory):
                            trade_to_close = trade_memory.pop(index - 1)
                            close_trade(trade_to_close)
                        else:
                            print("Invalid trade number.")
                    except ValueError:
                        print("Invalid input. Please enter a valid number.")
            elif option == "3":
                debug_clear_memory(trade_memory)
            else:
                print("Invalid option. Please choose a valid option.")
        
        cont = input("\nDo you want to continue? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting Trade Logger.")
            break

if __name__ == "__main__":
    main()
