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


def log_trade(ticker, trade_type, exp_date, strike_price, trade_number):
    """
    Logs a new trade for the selected ticker and trade type.
    Emits the trade event with provided parameters.
    """
    try:
        sio.connect(TRADE_SERVER_URL, transports=['websocket'])
        
        trade = {
            'ticker': ticker,
            'username': os.getenv('USERNAME', 'unknown'),
            'tradeType': trade_type,
            'expDate': exp_date,
            'strikePrice': strike_price,
            'status': 'open',
            'tradeNumber': trade_number
        }
        
        sio.emit('new_trade', trade)
        
        print(f"\nTrade logged successfully!")
        print(f"Ticker: {ticker}")
        print(f"Type: {trade_type}")
        print(f"Expiration: {exp_date}")
        print(f"Strike: {strike_price}")
        print(f"[trade #{trade_number}]")
        
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
        if 'tradeNumber' in trade:
            print(f"[trade #{trade['tradeNumber']}]")
        
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
    debug_message = "[DEBUG] Trade list memory cleared."
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


def select_trade_type():
    """
    Prompts the user to select a trade type using number 1 or 2.
    Returns 'CALL' or 'PUT'.
    """
    trade_type = None
    while trade_type is None:
        print("\nSelect Trade Type:")
        print("1. CALL")
        print("2. PUT")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            trade_type = "CALL"
        elif choice == "2":
            trade_type = "PUT"
        else:
            print("Invalid selection. Please enter 1 or 2.")
    return trade_type


def main():
    trade_memory = []  # Memory to store trades in progress
    trade_counter = 1  # Global trade counter
    
    # Print header and connection info BEFORE ticker selection.
    print("\n=== Trade Logger ===")
    print(f"Connecting to: {TRADE_SERVER_URL}")
    
    while True:
        if not trade_memory:
            # No trades in progress; get new trade details.
            ticker = select_ticker()
            trade_type = select_trade_type()
            exp_date = input("Enter expiration date (e.g., 2-14): ").strip()
            strike_price = input("Enter strike price: ").strip()
            
            if not all([exp_date, strike_price]):
                print("Error: All fields are required!")
                continue
            
            trade = log_trade(ticker, trade_type, exp_date, strike_price, trade_counter)
            if trade:
                trade_memory.append(trade)
                trade_counter += 1
        else:
            print("\nCurrent Trades in Memory:")
            for idx, trade in enumerate(trade_memory, start=1):
                print(f"{idx}. Ticker: {trade['ticker']}, Type: {trade['tradeType']}, "
                      f"Exp: {trade['expDate']}, Strike: {trade['strikePrice']}, "
                      f"Status: {trade['status']} [trade #{trade.get('tradeNumber', '?')}]")
            
            print("\nOptions:")
            print("1. Open New Trade")
            print("2. Close Trade")
            print("3. [DEBUG] CLEAR TRADE LIST MEMORY")
            option = input("Enter option number: ").strip()
            
            if option == "1":
                ticker = select_ticker()
                trade_type = select_trade_type()
                exp_date = input("Enter expiration date (e.g., 2-14): ").strip()
                strike_price = input("Enter strike price: ").strip()
                
                if not all([exp_date, strike_price]):
                    print("Error: All fields are required!")
                    continue
                
                trade = log_trade(ticker, trade_type, exp_date, strike_price, trade_counter)
                if trade:
                    trade_memory.append(trade)
                    trade_counter += 1
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
