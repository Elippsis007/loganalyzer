#!/usr/bin/env python3
"""
Deployment Setup Script for LogNarrator AI

This script helps prepare the project for deployment by:
1. Creating a clean project structure
2. Setting up Git repository
3. Providing deployment instructions
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def setup_git():
    """Initialize Git repository"""
    print("\n🚀 Setting up Git repository...")
    
    # Initialize git
    if not run_command("git init", "Initializing Git repository"):
        return False
    
    # Add all files
    if not run_command("git add .", "Adding files to Git"):
        return False
    
    # Make initial commit
    if not run_command('git commit -m "Initial commit: LogNarrator AI"', "Making initial commit"):
        return False
    
    print("\n✅ Git repository setup complete!")
    return True

def create_secrets_template():
    """Create a template for secrets"""
    secrets_dir = Path(".streamlit")
    secrets_dir.mkdir(exist_ok=True)
    
    secrets_template = """# Streamlit Secrets Template
# Copy this file to secrets.toml and add your actual API key
# DO NOT commit secrets.toml to version control

[secrets]
CLAUDE_API_KEY = "your-claude-api-key-here"
"""
    
    with open(secrets_dir / "secrets.toml.template", "w") as f:
        f.write(secrets_template)
    
    print("✅ Created secrets template at .streamlit/secrets.toml.template")

def main():
    """Main setup function"""
    print("🤖 LogNarrator AI - Deployment Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("ui_streamlit.py").exists():
        print("❌ Please run this script from the LogNarrator AI project directory")
        sys.exit(1)
    
    # Create secrets template
    create_secrets_template()
    
    # Setup Git
    if not setup_git():
        print("❌ Git setup failed")
        sys.exit(1)
    
    # Provide deployment instructions
    print("\n🚀 Deployment Instructions:")
    print("=" * 50)
    
    print("""
📋 Next Steps:

1. Push to GitHub:
   git remote add origin https://github.com/yourusername/loganalyzer.git
   git branch -M main
   git push -u origin main

2. Deploy on Streamlit Cloud:
   • Go to https://share.streamlit.io
   • Connect your GitHub account
   • Select your repository
   • Set main file: ui_streamlit.py
   • Add your API key in secrets management
   • Click Deploy

3. Alternative platforms:
   • Railway: https://railway.app
   • Render: https://render.com
   • Heroku: https://heroku.com

📄 See deploy.md for detailed instructions!
""")

if __name__ == "__main__":
    main() 