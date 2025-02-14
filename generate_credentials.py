import bcrypt
import json
import os
from getpass import getpass
from dotenv import load_dotenv

def generate_credentials():
    """Generate and store user credentials with encrypted password"""
    print("\n=== Credential Generator ===")
    
    # Get username
    while True:
        username = input("Enter username: ").strip()
        if username:
            break
        print("Username cannot be empty!")
    
    # Get password with confirmation
    while True:
        password = getpass("Enter password: ")
        if not password:
            print("Password cannot be empty!")
            continue
            
        confirm_password = getpass("Confirm password: ")
        if password != confirm_password:
            print("Passwords do not match! Please try again.")
            continue
        
        break
    
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    # Create credentials object
    credentials = {
        'username': username,
        'hashed_password': hashed_password.decode('utf-8')  # Convert bytes to string for storage
    }
    
    # Save to .env file
    with open('.env', 'a') as f:
        f.write(f"\nUSERNAME={credentials['username']}\n")
        f.write(f"HASHED_PASSWORD={credentials['hashed_password']}\n")
    
    print("\nCredentials generated and stored successfully!")
    print(f"Username: {username}")
    print("Password: [ENCRYPTED]")

if __name__ == "__main__":
    generate_credentials() 