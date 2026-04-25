# Badminton Video Detection

Badminton Video Detection is a Flask-based badminton analysis app that detects players in a video, tracks a selected player, estimates pose skeletons, and generates coaching feedback from posture and movement signals.

The current app supports:

- multi-person detection from the first frame
- selected-player tracking across the clip
- skeleton overlay drawing on detected players
- rule-based badminton coaching feedback
- longer structured "AI coach" responses
- follow-up chat about technique, posture, and movement

## Recommended App Folder

The clean runnable app lives in:

`badminton_analyzer_gpu_deployment/`

That folder contains the files needed to run or package the app:

- `app.py`
- `templates/`
- `static/`
- `Badminton_Analyzer_GPU_Colab.ipynb`
- `colab_setup.py`
- `yolov8n.pt`

If the repo also contains older nested folders under `content/`, use `badminton_analyzer_gpu_deployment/` as the source of truth for deployment.

## Project Layout

```text
badminton_analyzer_gpu_deployment/
  app.py
  static/
    styles.css
  templates/
    analysis.html
    base.html
    chat.html
    index.html
  Badminton_Analyzer_GPU_Colab.ipynb
  colab_setup.py
  yolov8n.pt
```

## Features

### 1. Player Detection

- Detects all visible people in the first frame
- Shows each detected person inside a labeled bounding box
- Draws full skeleton overlays when pose data is available

### 2. Player Tracking

- Lets the user choose which detected player to analyze
- Tracks that player across the video
- Extracts box movement metrics and skeleton-derived pose metrics

### 3. Skeleton Analysis

The app analyzes signals such as:

- trunk lean
- shoulder tilt
- hip tilt
- knee bend
- stance width
- centered balance
- athletic ready posture
- movement consistency

### 4. Coaching Output

The app returns:

- structured strengths and weaknesses
- suggested drills
- expected improvements
- longer formatted coach commentary
- follow-up chat responses

## Local Run

### Requirements

- Python 3.10+
- `torch`
- `torchvision`
- `flask`
- `flask-cors`
- `ultralytics`
- `transformers`
- `python-dotenv`
- OpenCV

### Install

From the clean app folder:

```bash
cd badminton_analyzer_gpu_deployment
pip install flask flask-cors ultralytics torch torchvision transformers python-dotenv opencv-python
```

### Start the app

```bash
cd badminton_analyzer_gpu_deployment
python app.py
```

Then open the local Flask URL shown in the terminal.

## Google Colab

The clean Colab-ready zip created from this project is:

`badminton_analyzer_fresh_colab.zip`

### Upload and unzip

```python
from google.colab import files
uploaded = files.upload()  # upload badminton_analyzer_fresh_colab.zip
```

```python
!rm -rf /content/badminton_analyzer
!mkdir -p /content/badminton_analyzer
!unzip -o badminton_analyzer_fresh_colab.zip -d /content/badminton_analyzer
%cd /content/badminton_analyzer
!ls
```

### Install dependencies

```python
!pip install flask ultralytics torch torchvision transformers flask-cors python-dotenv pyngrok
```

### Start Flask

```python
import threading
import time
from app import app

def run_app():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

thread = threading.Thread(target=run_app, daemon=True)
thread.start()

time.sleep(5)
print("Flask app running on port 5000")
```

### Expose with ngrok

```python
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(5000)
print(public_url)
```

## Notes

- The repo currently contains older duplicated folders and packaging artifacts. For deployment, prefer the clean folder and clean zip mentioned above.
- Local-only files such as `.env`, `uploads`, logs, and `__pycache__` should not be included in deployment zips.
- The pose analysis is designed to be helpful coaching feedback, not a medical or biomechanical diagnosis.

## Updated Files

The latest UI and coaching changes were made in:

- `content/badminton_analyzer/app.py`
- `content/badminton_analyzer/templates/analysis.html`
- `content/badminton_analyzer/templates/chat.html`

Those updates were synced into:

- `badminton_analyzer_gpu_deployment/app.py`
- `badminton_analyzer_gpu_deployment/templates/analysis.html`
- `badminton_analyzer_gpu_deployment/templates/chat.html`

## GitHub

Repository:

`https://github.com/Pohthato/Badminton-Video-Detection`
