#!/usr/bin/env python3
"""
Simple script to run the MCDM Learning Tool
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("Error: app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed. Please install requirements:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the Streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")

if __name__ == "__main__":
    main()
