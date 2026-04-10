#!/bin/bash
# Setup script to configure HuggingFace token for faster model downloads

echo "================================================"
echo "HuggingFace Token Configuration"
echo "================================================"
echo ""
echo "This will set your HF_TOKEN for faster model downloads."
echo "Your token will be stored in your system environment."
echo ""
read -p "Enter your HuggingFace token (from https://huggingface.co/settings/tokens): " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "Error: Token cannot be empty"
    exit 1
fi

# For Windows/Git Bash
export HF_TOKEN="$HF_TOKEN"
echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc

echo ""
echo "✓ Token configured successfully!"
echo "✓ Run 'source ~/.bashrc' or restart your terminal"
echo ""
echo "Your app will now use faster HF model downloads."
