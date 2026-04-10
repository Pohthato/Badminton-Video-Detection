#!/usr/bin/env python3
"""
Setup script to configure HuggingFace token for faster model downloads
Run this once to set your HF token in system environment
"""

import os
import sys

def setup_hf_token():
    print("=" * 50)
    print("HuggingFace Token Configuration")
    print("=" * 50)
    print()
    print("This will configure your HF_TOKEN for faster model downloads.")
    print("Get your token from: https://huggingface.co/settings/tokens")
    print()
    
    token = input("Enter your HuggingFace token: ").strip()
    
    if not token:
        print("✗ Error: Token cannot be empty")
        sys.exit(1)
    
    if len(token) < 20:
        print("✗ Error: Token seems too short (should be longer)")
        sys.exit(1)
    
    # Set environment variable
    os.environ['HF_TOKEN'] = token
    
    # For Windows - add to .env file
    env_file = '.env'
    with open(env_file, 'w') as f:
        f.write(f"HF_TOKEN={token}\n")
    
    print()
    print("✓ Token configured successfully!")
    print(f"✓ Saved to {env_file}")
    print()
    print("Your app will now use faster HF model downloads.")
    print()
    print("Note: If you set HF_TOKEN in terminal before running app.py:")
    print("  set HF_TOKEN=your_token_here")
    print("  python app.py")

if __name__ == '__main__':
    setup_hf_token()
