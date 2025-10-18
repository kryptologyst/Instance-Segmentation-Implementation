#!/usr/bin/env python3
"""
Setup script for the instance segmentation project.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Instance Segmentation Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data/samples", "data/synthetic", "outputs", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Generate sample data
    if not run_command("python src/data_generator.py", "Generating sample data"):
        print("⚠️  Sample data generation failed, but continuing...")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Launch the web interface: streamlit run web_app/app.py")
    print("2. Or use the CLI: python cli.py --help")
    print("3. Check the README.md for more information")
    
    print("\n📁 Project structure:")
    print("├── src/                    # Source code")
    print("├── web_app/               # Streamlit web interface")
    print("├── config/                # Configuration files")
    print("├── data/                  # Data directory")
    print("├── models/                # Model storage")
    print("├── tests/                 # Test files")
    print("├── cli.py                 # Command-line interface")
    print("├── requirements.txt      # Dependencies")
    print("└── README.md              # Documentation")


if __name__ == "__main__":
    main()
