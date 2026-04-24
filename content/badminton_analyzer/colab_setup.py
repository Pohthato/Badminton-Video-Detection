# Badminton Analyzer - GPU Colab Version
# Upload these files to Colab:
# 1. app.py (your main application)
# 2. yolov8n.pt (YOLO model weights)
# 3. templates/index.html
# 4. templates/analysis.html
# 5. templates/chat.html
# 6. templates/base.html
# 7. static/styles.css
# 8. Your test video file (e.g., badminton_game.mp4)

# Install dependencies
!pip install flask ultralytics torch torchvision transformers flask-cors python-dotenv

# Mount Google Drive (optional, for larger files)
from google.colab import drive
drive.mount('/content/drive')

# Create directories
!mkdir -p templates static uploads

# Upload your files here, then run:
import os
os.chdir('/content')

# Enable GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Run your app
# !python app.py