"""
Setup script for MCDM Learning Tool
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def main():
    """Main setup function"""
    print("🎯 MCDM Learning Tool Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        sys.exit(1)
    
    print("\n🚀 Setup complete!")
    print("\nTo run the application:")
    print("  python run.py")
    print("  or")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()
