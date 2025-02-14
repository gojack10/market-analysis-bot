import socketio
import os
from dotenv import load_dotenv
import sys
import time  # Add time import

# Load environment variables
load_dotenv()

# Initialize SocketIO client
sio = socketio.Client()

# Get server URL from environment variable
TRADE_SERVER_URL = os.getenv('TRADE_SERVER_URL', 'http://localhost:8000')

def log_trade():
    try:
        # Connect to the server with explicit transport setting
        sio.connect(TRADE_SERVER_URL, transports=['websocket'])
        
        # Get trade details from user
        print("\n=== Trade Logger ===")
        print(f"Connecting to: {TRADE_SERVER_URL}")
        trade_type = input("Enter trade type (call/put): ").strip().upper()
        exp_date = input("Enter expiration date (e.g., 2-14): ").strip()
        strike_price = input("Enter strike price: ").strip()
        
        # Validate inputs
        if not all([trade_type, exp_date, strike_price]):
            print("Error: All fields are required!")
            return
        
        if trade_type not in ['CALL', 'PUT']:
            print("Error: Trade type must be either 'call' or 'put'!")
            return
            
        # Emit the trade
        sio.emit('new_trade', {
            'username': os.getenv('USERNAME', 'unknown'),
            'tradeType': trade_type,
            'expDate': exp_date,
            'strikePrice': strike_price
        })
        
        print(f"\nTrade logged successfully!")
        print(f"Type: {trade_type}")
        print(f"Expiration: {exp_date}")
        print(f"Strike: {strike_price}")
        
        # Add delay to ensure event propagation
        time.sleep(1)
        
    except Exception as e:
        print(f"Error connecting to {TRADE_SERVER_URL}: {e}")
        print("Make sure the trade server is running and the TRADE_SERVER_URL is correct")
    
    finally:
        if sio.connected:
            sio.disconnect()

if __name__ == "__main__":
    log_trade() 