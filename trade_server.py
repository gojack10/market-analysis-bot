from flask import Flask
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# Initialize SocketIO with CORS allowed
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@socketio.on("new_trade")
def handle_new_trade(data):
    """Handle new trade broadcasts."""
    try:
        message = (
            f"{data.get('username', 'unknown')} NEW TRADE - "
            f"{data.get('ticker', 'unknown')} {data.get('tradeType', '')} "
            f"{data.get('expDate', '')} @ {data.get('strikePrice', '')}"
        )
        if 'tradeNumber' in data:
            message += f" [trade #{data['tradeNumber']}]"
        print(message)  # Log to server console
        emit("new_trade", message, broadcast=True)  # Broadcast to all connected clients
    except Exception as e:
        print(f"Error handling trade: {e}")
        emit("new_trade", "ERROR: Failed to process trade.")

@socketio.on("close_trade")
def handle_close_trade(data):
    """Handle trade close events."""
    try:
        message = (
            f"{data.get('username', 'unknown')} TRADE CLOSED - "
            f"{data.get('ticker', 'unknown')} {data.get('tradeType', '')} "
            f"{data.get('expDate', '')} @ {data.get('strikePrice', '')}"
        )
        if 'tradeNumber' in data:
            message += f" [trade #{data['tradeNumber']}]"
        print(message)
        emit("close_trade", message, broadcast=True)
    except Exception as e:
        print(f"Error handling close trade: {e}")
        emit("close_trade", "ERROR: Failed to process close trade.")

@socketio.on("debug_log")
def handle_debug_log(data):
    """Handle debug log events."""
    try:
        message = data.get('message', '')
        print(message)
        emit("debug_log", message, broadcast=True)
    except Exception as e:
        print(f"Error handling debug log: {e}")
        emit("debug_log", "ERROR: Failed to process debug log.")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connect_response', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
