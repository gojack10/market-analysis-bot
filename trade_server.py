from flask import Flask
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Initialize SocketIO with CORS allowed
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("new_trade")
def handle_new_trade(data):
    """Handle new trade broadcasts."""
    try:
        message = f"{data.get('username', 'unknown')} NEW TRADE - {data.get('tradeType', '')} {data.get('expDate','')} @ {data.get('strikePrice','')}"
        print(message)  # Log to server console
        emit("new_trade", message, broadcast=True)  # Broadcast to all connected clients
    except Exception as e:
        print(f"Error handling trade: {e}")
        emit("new_trade", "ERROR: Failed to process trade.")

if __name__ == "__main__":
    print("Trade server running on http://0.0.0.0:8000")
    print("Waiting for trades from log_trade.py...")
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, use_reloader=False) 