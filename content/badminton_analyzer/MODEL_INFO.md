# Badminton Video Detection Analyzer

AI-powered badminton player detection, analysis, and coaching feedback system.

## Features

✅ **YOLO-based Player Detection** - Real-time detection with NMS filtering
✅ **Multi-step Workflow** - Upload → Analyze → Chat flow
✅ **ChatGPT-style Coach** - Interactive coaching feedback with loading animations
✅ **Fast Model** - Uses `flan-t5-small` (~3x faster than base)
✅ **Modern UI** - Dark theme with smooth animations

## Model Information

**Current Model:** `google/flan-t5-small`
- **Speed:** ~500ms per response (3x faster than flan-t5-base)
- **Quality:** Good coaching feedback for badminton
- **Memory:** Low footprint, optimized for CPU

**Alternative Models:**
- `google/flan-t5-base` - Better quality, slower (~1.5s)
- `google/flan-t5-large` - Best quality, much slower (~3s+)
- `distilbert-base-uncased` - Maximum speed (~200ms), shorter responses

## Setup

### 1. Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate      # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install flask flask-cors opencv-python numpy torch ultralytics transformers mediapipe
```

### 3. Configure HuggingFace Token (Optional but Recommended)

For faster model downloads, set your HF token:

```bash
python setup_hf_token.py
```

Or manually set:
```bash
set HF_TOKEN=your_token_here  # Windows
export HF_TOKEN=your_token_here  # macOS/Linux
```

### 4. Run the App
```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

## Usage

1. **Upload Video** (Step 1) - Select a badminton match video
2. **Detect Players** - YOLO identifies players with improved filtering
3. **Analyze** (Step 2) - View detailed pose and movement metrics
4. **Get Coaching** (Step 3) - Ask the AI coach questions in real-time

## Performance

- **Upload Processing:** ~2-5 seconds per video
- **Coach Response:** ~0.5 seconds (with flan-t5-small)
- **Player Detection:** Real-time with 30 FPS video

## Recent Improvements

- ✅ Upgraded to `flan-t5-small` for 3x faster responses
- ✅ NMS filtering to eliminate ghost player detections
- ✅ ChatGPT-style chat interface
- ✅ Loading animations for all async operations
- ✅ Suppressed HuggingFace warnings
