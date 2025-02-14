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
        # Build message including the ticker so that it logs as: 
        # "jack NEW TRADE - NVDA CALL 2-14 @ 205"
        message = (
            f"{data.get('username', 'unknown')} NEW TRADE - "
            f"{data.get('ticker', 'unknown')} {data.get('tradeType', '')} "
            f"{data.get('expDate', '')} @ {data.get('strikePrice', '')}"
        )
        print(message)  # Log to server console
        emit("new_trade", message, broadcast=True)  # Broadcast to all connected clients
    except Exception as e:
        print(f"Error handling trade: {e}")
        emit("new_trade", "ERROR: Failed to process trade.")

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
