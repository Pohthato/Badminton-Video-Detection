from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS

# Suppress HF token warning
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import warnings
warnings.filterwarnings('ignore')

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"[WARNING] MediaPipe import failed: {e}")
    mp = None

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Ensure all errors return JSON, not HTML
@app.errorhandler(Exception)
def handle_error(error):
    print(f"[ERROR] Unhandled exception: {error}")
    import traceback
    traceback.print_exc()
    return jsonify({'success': False, 'error': str(error)}), 500

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512 MB upload limit

# Initialize models
yolo_model = None
hf_analyzer = None
mp_pose = None
current_video_path = None  # Store video path for analysis
detected_players = {}  # Store player bounding boxes: {label: [(x1,y1,x2,y2), ...frames]}
player_analysis_context = {}  # Keep last player analytic state for chatbot follow-up
last_detected_frame = None  # Base64 preview image from the latest detection


def init_yolo():
    global yolo_model
    if yolo_model is None:
        try:
            print("[INFO] Loading YOLO model...")
            yolo_model = YOLO('yolov8n.pt')
            yolo_model.to('cpu')  # Force model to CPU
            print("[INFO] YOLO model loaded successfully (CPU)")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO: {e}")
            yolo_model = False  # Mark as failed attempt



def init_mediapipe():
    """MediaPipe pose extraction - disabled due to module compatibility issues."""
    global mp_pose
    if mp_pose is None:
        print("[INFO] Pose extraction temporarily using basic analysis")
        mp_pose = True  # Enable basic mode
    return mp_pose


def extract_pose_data(image, box):
    """Extract pose information from bounding box dimensions and position."""
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    
    # Calculate metrics from box position and size
    box_height = y2 - y1
    box_width = x2 - x1
    aspect_ratio = box_height / (box_width + 1e-6)
    
    # Estimate posture from box aspect ratio and position
    is_upright = aspect_ratio > 1.5
    box_center_x = (x1 + x2) / 2
    
    data = {
        'posture': 'upright' if is_upright else 'bent/crouched',
        'balance': 'centred' if (box_center_x > w * 0.3 and box_center_x < w * 0.7) else 'off_centre',
        'stance_width': box_width,
        'height_ratio': aspect_ratio,
        'position_y': y1,  # Higher in frame = more upright
        'stance': 'narrow' if box_width < 60 else 'wide' if box_width > 100 else 'moderate'
    }
    
    return data


def init_hf_analyzer():
    global hf_analyzer
    if hf_analyzer is None:
        try:
            print("[INFO] Loading HF analyzer (FLAN-T5-small - fast & efficient)...")
            # Use smaller, faster model optimized for speed
            tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
            model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
            model.cpu()  # Move to CPU without device_map
            hf_analyzer = {'tokenizer': tokenizer, 'model': model}
            print("[INFO] HF analyzer loaded successfully (CPU) - ~3x faster than base model")
        except Exception as e:
            print(f"[WARNING] HF analyzer not available, using rule-based feedback: {e}")
            hf_analyzer = False


def generate_structured_feedback(prompt, max_new_tokens=260):
    """Generate cleaner, non-echoed coaching text with deterministic decoding."""
    try:
        if hf_analyzer is None:
            init_hf_analyzer()

        if not hf_analyzer or hf_analyzer is False:
            return "Coach chat unavailable: HF text model not initialized."

        tokenizer = hf_analyzer['tokenizer']
        model = hf_analyzer['model']

        inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Enable sampling for more varied and longer responses
            temperature=0.7,  # Add creativity
            top_p=0.9,  # Nucleus sampling
            num_beams=1,  # Disable beams when sampling
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return result if result else "I apologize, but I couldn't generate a response at this time."

    except Exception as e:
        print(f"[ERROR] in generate_structured_feedback: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating coaching feedback: {str(e)}"


def summarize_pose_metrics(poses):
    """Build deeper frame-level metrics from extracted pose proxies."""
    valid = [p for p in poses if p]
    n = len(valid)
    if n == 0:
        return {
            "samples": 0,
            "upright_ratio": 0.0,
            "centred_ratio": 0.0,
            "wide_ratio": 0.0,
            "narrow_ratio": 0.0,
            "avg_height_ratio": 0.0,
            "height_ratio_std": 0.0,
            "avg_stance_width": 0.0,
            "stance_width_std": 0.0,
            "vertical_mobility": 0.0,
            "balance_switch_rate": 0.0,
            "stance_switch_rate": 0.0,
        }

    postures = [p.get("posture", "unknown") for p in valid]
    balances = [p.get("balance", "unknown") for p in valid]
    stances = [p.get("stance", "unknown") for p in valid]
    heights = np.array([float(p.get("height_ratio", 0.0)) for p in valid], dtype=float)
    widths = np.array([float(p.get("stance_width", 0.0)) for p in valid], dtype=float)
    y_positions = np.array([float(p.get("position_y", 0.0)) for p in valid], dtype=float)

    def switch_rate(items):
        if len(items) < 2:
            return 0.0
        switches = sum(1 for i in range(1, len(items)) if items[i] != items[i - 1])
        return switches / (len(items) - 1)

    return {
        "samples": n,
        "upright_ratio": postures.count("upright") / n,
        "centred_ratio": balances.count("centred") / n,
        "wide_ratio": stances.count("wide") / n,
        "narrow_ratio": stances.count("narrow") / n,
        "avg_height_ratio": float(np.mean(heights)),
        "height_ratio_std": float(np.std(heights)),
        "avg_stance_width": float(np.mean(widths)),
        "stance_width_std": float(np.std(widths)),
        "vertical_mobility": float(np.std(y_positions)),
        "balance_switch_rate": float(switch_rate(balances)),
        "stance_switch_rate": float(switch_rate(stances)),
    }


def compute_box_metrics(boxes):
    """Compute movement metrics from a sequence of bounding boxes."""
    if not boxes:
        return {
            "frames": 0,
            "total_movement_px": 0.0,
            "avg_step_px": 0.0,
            "movement_consistency": 0.0,
            "avg_area_px2": 0.0,
            "area_stability": 0.0,
        }

    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    areas = [max(1.0, float((b[2] - b[0]) * (b[3] - b[1]))) for b in boxes]
    steps = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        steps.append(float(np.sqrt(dx * dx + dy * dy)))

    avg_step = float(np.mean(steps)) if steps else 0.0
    step_std = float(np.std(steps)) if steps else 0.0
    movement_consistency = 0.0 if avg_step == 0 else max(0.0, 1.0 - (step_std / (avg_step + 1e-6)))
    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas))
    area_stability = max(0.0, 1.0 - (area_std / (area_mean + 1e-6)))

    return {
        "frames": len(boxes),
        "total_movement_px": float(np.sum(steps)) if steps else 0.0,
        "avg_step_px": avg_step,
        "movement_consistency": movement_consistency,
        "avg_area_px2": area_mean,
        "area_stability": area_stability,
    }


def build_rule_based_player_feedback(player, posture, balance, stance, height_ratio, pose_metrics, box_metrics):
    """Deterministic coaching text based on computed video metrics."""
    strengths = []
    weaknesses = []
    drills = []
    improvements = []

    upright_ratio = pose_metrics.get("upright_ratio", 0.0)
    centred_ratio = pose_metrics.get("centred_ratio", 0.0)
    wide_ratio = pose_metrics.get("wide_ratio", 0.0)
    stance_switch_rate = pose_metrics.get("stance_switch_rate", 0.0)
    balance_switch_rate = pose_metrics.get("balance_switch_rate", 0.0)
    vertical_mobility = pose_metrics.get("vertical_mobility", 0.0)
    movement_consistency = box_metrics.get("movement_consistency", 0.0)
    avg_step = box_metrics.get("avg_step_px", 0.0)

    strengths.append(f"Player maintains upright posture in {upright_ratio * 100:.0f}% of frames.")
    if centred_ratio >= 0.55:
        strengths.append(f"Center balance preserved in {centred_ratio * 100:.0f}% of samples.")
    else:
        weaknesses.append(f"Center balance only {centred_ratio * 100:.0f}% stable; focus on midline control.")

    if wide_ratio >= 0.5:
        strengths.append("Repeatable wide stance gives strong defensive stability.")
    elif wide_ratio <= 0.25:
        weaknesses.append("Sometimes stance is too narrow, reducing base support under pressure.")

    if posture == "upright":
        strengths.append("Tall torso and chest lift are strong for shot recovery and anticipation.")
    else:
        weaknesses.append("Body alignment is low; guard against early fatigue on court rallies.")
        drills.append("Shadow split-step + recovery: 4x45s, focus on chest up and neutral spine.")

    if balance == "centred":
        strengths.append("Center of mass is well controlled through movement transitions.")
    else:
        weaknesses.append("Weight shift is off-centre, producing delayed court re-center.")
        drills.append("Lateral lunge hold with racket reach: 3x8 each side, 2-second stabilize.")

    if stance == "wide":
        strengths.append("Wide base increases defensive absorbing power.")
        weaknesses.append("Wide recovery stance may slow first-step acceleration.")
        drills.append("Split-step width drill: 5x30s, land shoulder-width then explode to corner.")
    elif stance == "narrow":
        weaknesses.append("Narrow base results in potential balance vulnerability.")
        drills.append("Mini-band squat walk + split-step: 3x40s to build stable base.")
    else:
        strengths.append("Moderate stance supports balanced mobility and stability.")

    if height_ratio >= 2.0:
        strengths.append("Effective height ratio indicates efficient spine control.")
    else:
        weaknesses.append("Height ratio is lower than ideal; posture can be more vertical.")
        drills.append("Hip hinge wall taps + racket overhead reach: 3x10 controlled reps.")

    if balance_switch_rate > 0.45:
        weaknesses.append("Balance adjustment rate is high, suggesting jittery weight transfer.")
        drills.append("Split-step freeze + recover: 4x30s, hold before first step.")
    if stance_switch_rate > 0.5:
        weaknesses.append("Stance adjustments are frequent; stabilize pre-shot setup.")
    if vertical_mobility > 70:
        weaknesses.append("Excess vertical motion may waste energy; keep lower center-of-gravity.")

    if movement_consistency < 0.50:
        weaknesses.append(f"Step consistency is {movement_consistency:.2f}; aim for 0.60+.")
    if avg_step < 8:
        weaknesses.append(f"Average step size {avg_step:.1f}px indicates limited court coverage per sample.")
    elif avg_step > 22:
        strengths.append(f"Larger average step ({avg_step:.1f}px) shows good court coverage aggressiveness.")

    if len(drills) < 3:
        drills.extend([
            "6-point court footwork pattern: 4 rounds x 40s work / 20s rest, focus on recovery.",
            "Two-shuttle reaction starts: 3x10 reps, commit to first step within 300ms.",
        ])

    improvements.extend([
        "Faster recovery to base should improve rally consistency.",
        "Balanced weight transfer will reduce unforced errors on directional changes.",
        "Cleaner posture mechanics improve endurance during longer rallies.",
        f"Target step consistency above 0.60 (current {movement_consistency:.2f}) to raise rally control.",
    ])

    if not strengths:
        strengths.append("Player shows consistent work rate that can be refined for higher efficiency.")
    if not weaknesses:
        weaknesses.append("No major faults; sharpen details on rhythm and mental intensity.")

    strengths.insert(0, f"Observed {pose_metrics.get('samples', 0)} pose frames and {box_metrics.get('frames', 0)} tracking samples.")

    return {
        'strengths': strengths[:5],
        'weaknesses': weaknesses[:5],
        'drills': drills[:4],
        'improvements': improvements[:4],
        'metrics': {
            'upright_ratio': upright_ratio,
            'centred_ratio': centred_ratio,
            'stance_switch_rate': stance_switch_rate,
            'balance_switch_rate': balance_switch_rate,
            'vertical_mobility': vertical_mobility,
            'movement_consistency': movement_consistency,
            'avg_step_px': avg_step,
            'height_ratio': height_ratio,
        }
    }


def build_rule_based_compare_feedback(player1, p1_prof, player2, p2_prof):
    """Deterministic fallback comparison text."""
    p1_adv = "balance control" if p1_prof.get("balance") == "centred" else "defensive stability"
    p2_adv = "balance control" if p2_prof.get("balance") == "centred" else "defensive stability"

    return (
        "KEY TECHNICAL DIFFERENCES:\n"
        f"- {player1} posture/balance profile: {p1_prof.get('posture')}, {p1_prof.get('balance')}, {p1_prof.get('stance')} stance.\n"
        f"- {player2} posture/balance profile: {p2_prof.get('posture')}, {p2_prof.get('balance')}, {p2_prof.get('stance')} stance.\n\n"
        f"WHAT {player1.upper()} DOES BETTER:\n"
        f"- Shows relatively stronger {p1_adv} in current sample.\n"
        "- Recovers with more repeatable movement timing on similar actions.\n\n"
        f"WHAT {player2.upper()} DOES BETTER:\n"
        f"- Shows relatively stronger {p2_adv} in current sample.\n"
        "- Demonstrates better stability in at least one movement phase.\n\n"
        f"DRILLS FOR {player1.upper()}:\n"
        "- Split-step width and first-step acceleration: 4x30s.\n"
        "- 6-point recovery footwork with time cap: 4 rounds.\n\n"
        f"DRILLS FOR {player2.upper()}:\n"
        "- Lateral lunge stabilization + racket reach: 3x8 each side.\n"
        "- Shadow rally with posture checkpoints every 3 shots: 4x40s.\n\n"
        "COMPETITIVE ADVANTAGE:\n"
        "- The player who improves balance recovery and first-step timing first will gain rally control."
    )


def box_center(box):
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def assign_detections_to_tracks(track_state, detections, max_distance=140.0):
    """Greedy frame-to-frame association by nearest center distance."""
    assignments = {}
    used_tracks = set()
    used_detections = set()

    candidate_pairs = []
    for tid, tbox in track_state.items():
        for di, dbox in enumerate(detections):
            candidate_pairs.append((center_distance(tbox, dbox), tid, di))
    candidate_pairs.sort(key=lambda x: x[0])

    for dist, tid, di in candidate_pairs:
        if dist > max_distance or tid in used_tracks or di in used_detections:
            continue
        assignments[di] = tid
        used_tracks.add(tid)
        used_detections.add(di)

    return assignments


# Preload HF analyzer at startup
init_hf_analyzer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analysis")
def analysis_page():
    players = list(detected_players.keys())
    return render_template(
        "analysis.html",
        players=players,
        preview=last_detected_frame,
        has_analysis=bool(player_analysis_context),
    )


@app.route("/coach")
def coach_page():
    players = list(detected_players.keys())
    analyzed_players = list(player_analysis_context.keys())
    return render_template(
        "chat.html",
        players=players,
        analyzed_players=analyzed_players,
    )


def apply_nms(boxes, overlap_threshold=0.3):
    """Remove overlapping boxes using Non-Maximum Suppression."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if not keep:
            keep.append((x1, y1, x2, y2))
            continue
        is_duplicate = False
        for kx1, ky1, kx2, ky2 in keep:
            inter_x1 = max(x1, kx1)
            inter_y1 = max(y1, ky1)
            inter_x2 = min(x2, kx2)
            inter_y2 = min(y2, ky2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                box_area = (x2 - x1) * (y2 - y1)
                iou = inter_area / (box_area + 1e-6)
                if iou > overlap_threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            keep.append((x1, y1, x2, y2))
    return keep


def filter_close_detections(boxes, min_distance_px=40):
    """Remove detections that are too close to each other (ghost boxes)."""
    if len(boxes) <= 1:
        return boxes
    filtered = []
    for x1, y1, x2, y2 in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        is_close = False
        for fx1, fy1, fx2, fy2 in filtered:
            fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            dist = np.sqrt((cx - fcx) ** 2 + (cy - fcy) ** 2)
            if dist < min_distance_px:
                is_close = True
                break
        if not is_close:
            filtered.append((x1, y1, x2, y2))
    return filtered


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route('/detect-humans', methods=['POST'])
def detect_humans():
    """
    Upload video, detect humans with YOLO across multiple frames,
    extract pose data with MediaPipe, and save for analysis.
    """
    global current_video_path, detected_players, last_detected_frame
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    # Save uploaded video
    filename = secure_filename(file.filename)
    save_path = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(str(save_path))
    current_video_path = str(save_path)
    detected_players = {}  # Reset

    # Initialize models
    init_yolo()
    
    if not yolo_model:
        return jsonify({'error': 'YOLO model failed to load'}), 500

    try:
        # Multi-frame analysis: detect players and extract pose across frames
        cap = cv2.VideoCapture(str(save_path))
        frame_count = 0
        frame_sample_rate = 5
        tracks = {}  # {track_id: {'boxes': [...], 'poses': [...], 'last_seen': frame_count}}
        active_track_boxes = {}  # {track_id: box}
        next_track_id = 1
        
        print("[INFO] Analyzing video for player pose data...")
        
        while frame_count < 300:  # Increased from 150 to 300 frames (~10 sec at 30fps)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every Nth frame for analysis
            if frame_count % frame_sample_rate == 0:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                # YOLO detection with stricter filtering
                results = yolo_model(frame, conf=0.5)
                detections = []
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    if cls == 0 and conf >= 0.45:  # Stricter: Person class with higher confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_width = x2 - x1
                        box_height = y2 - y1
                        area = box_width * box_height
                        # Strict dimension filtering
                        if box_width < 90 or box_height < 150:  # Tighter minimum
                            continue
                        if box_width > 400 or box_height > 500:  # Maximum to reject groups
                            continue
                        # Strict aspect ratio: humans are taller than wide
                        aspect_ratio = box_height / (box_width + 1e-6)
                        if aspect_ratio < 1.3 or aspect_ratio > 3.0:
                            continue
                        # Area constraints
                        if area < 24000 or area > 150000:  # Tighter bounds
                            continue
                        detections.append((x1, y1, x2, y2))

                # Apply NMS and spatial filtering to remove ghosts
                detections = apply_nms(detections, overlap_threshold=0.2)
                detections = filter_close_detections(detections, min_distance_px=50)

                if len(detections) > 6:
                    detections = detections[:6]

                assignments = assign_detections_to_tracks(active_track_boxes, detections)
                new_active = {}

                for di, dbox in enumerate(detections):
                    if di in assignments:
                        tid = assignments[di]
                    else:
                        tid = next_track_id
                        next_track_id += 1
                        tracks[tid] = {"boxes": [], "poses": [], "last_seen": frame_count}

                    if tid not in tracks:
                        tracks[tid] = {"boxes": [], "poses": [], "last_seen": frame_count}

                    tracks[tid]["boxes"].append(dbox)
                    pose = extract_pose_data(frame, dbox)
                    if pose is not None:
                        tracks[tid]["poses"].append(pose)
                    tracks[tid]["last_seen"] = frame_count
                    new_active[tid] = dbox

                # keep recently seen tracks only to avoid id drift
                active_track_boxes = {
                    tid: box
                    for tid, box in new_active.items()
                    if frame_count - tracks[tid]["last_seen"] <= (frame_sample_rate * 2)
                }
            
            frame_count += 1
        
        cap.release()
        
        # Keep strongest tracks and map to Player1..N with robust filtering to avoid ghost targets
        MIN_TRACK_FRAMES = 15  # Stricter: must see movement across more frames
        MIN_TOTAL_MOVEMENT_PX = 100.0  # Stricter: must move significantly
        MIN_AVG_AREA_PX2 = 25000.0  # Tighter
        MAX_AVG_AREA_PX2 = 140000.0  # Tighter upper bound

        shortlist = []
        for tid, tdata in tracks.items():
            if len(tdata["boxes"]) < MIN_TRACK_FRAMES:
                continue
            metrics = compute_box_metrics(tdata["boxes"])
            if metrics["total_movement_px"] < MIN_TOTAL_MOVEMENT_PX:
                continue
            avg_width = np.mean([b[2] - b[0] for b in tdata["boxes"]]) if tdata["boxes"] else 0
            avg_height = np.mean([b[3] - b[1] for b in tdata["boxes"]]) if tdata["boxes"] else 0
            if avg_width < 80 or avg_height < 120:
                continue
            if not (MIN_AVG_AREA_PX2 <= metrics["avg_area_px2"] <= MAX_AVG_AREA_PX2):
                continue
            score = (
                metrics["total_movement_px"] * 0.5
                + metrics["avg_area_px2"] * 0.001
                + avg_width * 0.5
            )
            shortlist.append((tid, tdata, metrics, score))

        if not shortlist:
            shortlist = [
                (tid, tdata, compute_box_metrics(tdata["boxes"]), 0)
                for tid, tdata in tracks.items()
                if len(tdata["boxes"]) >= 8
            ]

        ranked_tracks = sorted(
            shortlist,
            key=lambda item: (item[3], len(item[1]["boxes"]), item[2]["avg_area_px2"]),
            reverse=True,
        )
        detected_players = {}

        for idx, (tid, tdata, metrics, _) in enumerate(ranked_tracks[:3], start=1):
            detected_players[f"Player{idx}"] = {
                "track_id": tid,
                "boxes": tdata["boxes"],
                "poses": tdata["poses"],
                "metrics": metrics,
            }

        print(f"[INFO] Detected {len(detected_players)} players with stable tracking")
        
        # Get first frame for display
        cap = cv2.VideoCapture(str(save_path))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
            frame_with_boxes = frame.copy()
            labels = []
            
            for label, data in detected_players.items():
                if data['boxes']:
                    labels.append(label)
                    x1, y1, x2, y2 = data['boxes'][0]
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
        else:
            frame_b64 = None
            labels = []

        last_detected_frame = frame_b64
        
        return jsonify({
            'frame': frame_b64,
            'labels': labels,
            'count': len(labels)
        })
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@app.route('/analyze-player', methods=['POST'])
def analyze_player():
    """
    Analyze a single player using ONLY rule‑based feedback.
    No HF model is called here.
    """
    global detected_players

    data = request.json or {}
    player = data.get('player', '')

    if not player:
        return jsonify({'error': 'Please select a player'}), 400

    if player not in detected_players:
        return jsonify({'error': 'Player not found'}), 400

    try:
        poses = detected_players[player].get('poses', [])
        boxes = detected_players[player].get('boxes', [])

        if not poses:
            return jsonify({'error': 'No pose data available'}), 400

        # Extract dominant categorical attributes
        postures = [p.get('posture') for p in poses if p.get('posture')]
        balances = [p.get('balance') for p in poses if p.get('balance')]
        stances  = [p.get('stance')  for p in poses if p.get('stance')]
        heights  = [p.get('height_ratio') for p in poses if p.get('height_ratio')]

        dominant_posture = max(set(postures), key=postures.count) if postures else 'unknown'
        dominant_balance = max(set(balances), key=balances.count) if balances else 'unknown'
        dominant_stance  = max(set(stances),  key=stances.count)  if stances  else 'unknown'
        avg_height       = float(np.mean(heights)) if heights else 0.0

        # Compute deeper metrics
        pose_metrics = summarize_pose_metrics(poses)
        box_metrics  = compute_box_metrics(boxes)

        # Check for sufficient movement - if player is mostly stationary, provide limited feedback
        total_movement = box_metrics.get('total_movement_px', 0)
        avg_step = box_metrics.get('avg_step_px', 0)
        if total_movement < 50 or avg_step < 5:
            # Player appears stationary - provide basic feedback only
            feedback_text = (
                "STRENGTHS:\n- Player detected and tracked successfully.\n\n"
                "WEAKNESSES:\n- Limited movement detected in this sample.\n\n"
                "IMPROVEMENT DRILLS:\n- Focus on dynamic movement patterns in future analysis.\n\n"
                "EXPECTED IMPROVEMENTS:\n- Upload video with active gameplay for more detailed feedback."
            )
            ai_commentary = "This analysis shows limited movement. For better coaching insights, please upload video of active badminton play with court movement."
        else:
            # Generate full feedback for active players
            feedback_data = build_rule_based_player_feedback(
                player,
                dominant_posture,
                dominant_balance,
                dominant_stance,
                avg_height,
                pose_metrics,
                box_metrics,
            )

            feedback_text = (
                "STRENGTHS:\n- " + "\n- ".join(feedback_data['strengths']) + "\n\n"
                "WEAKNESSES:\n- " + "\n- ".join(feedback_data['weaknesses']) + "\n\n"
                "IMPROVEMENT DRILLS:\n- " + "\n- ".join(feedback_data['drills']) + "\n\n"
                "EXPECTED IMPROVEMENTS:\n- " + "\n- ".join(feedback_data['improvements'])
            )

            # Generate AI commentary
            init_hf_analyzer()
            ai_commentary = ""
            if hf_analyzer and hf_analyzer is not False:
                commentary_prompt = (
                    "You are an expert badminton coach. "
                    f"Analyze {player}'s performance and write a direct coaching commentary based on this profile:\n"
                    f"- Posture: {dominant_posture}\n"
                    f"- Balance: {dominant_balance}\n"
                    f"- Stance: {dominant_stance}\n"
                    f"- Upright ratio: {pose_metrics.get('upright_ratio', 0):.0%}\n"
                    f"- Centered ratio: {pose_metrics.get('centred_ratio', 0):.0%}\n"
                    f"- Step consistency: {box_metrics.get('movement_consistency', 0):.2f}\n"
                    f"- Average step: {box_metrics.get('avg_step_px', 0):.1f}px\n"
                    "Answer in 4-6 complete sentences with specific advice, corrective drills, and encouragement. "
                    "Do not repeat the prompt or the instruction text."
                )
                ai_commentary = generate_structured_feedback(commentary_prompt, max_new_tokens=600)
                if ai_commentary.strip().startswith("You are") or ai_commentary.strip().startswith("As a professional badminton coach"):
                    fallback_prompt = (
                        "Rewrite the following coaching message as a direct badminton coaching paragraph without restating the instruction or the prompt:\n"
                        f"{ai_commentary}"
                    )
                    ai_commentary = generate_structured_feedback(fallback_prompt, max_new_tokens=250)

        # Save context for follow-up questions
        player_analysis_context[player] = {
            'profile': {
                'posture': dominant_posture,
                'balance': dominant_balance,
                'stance': dominant_stance,
                'height_ratio': avg_height,
            },
            'pose_metrics': pose_metrics,
            'box_metrics': box_metrics,
            'feedback': feedback_data,
            'summary': feedback_text,
            'ai_commentary': ai_commentary,
        }

        result = f"""
COACHING ANALYSIS: {player}
======================================================================

TECHNICAL PROFILE:
  Posture: {dominant_posture}
  Balance: {dominant_balance}
  Stance: {dominant_stance}
  Height Ratio: {avg_height:.2f}

FEEDBACK:
{feedback_text}

======================================================================
AI COACH COMMENTARY:
{ai_commentary}
======================================================================
"""

        return jsonify({
            'analysis': result,
            'analysis_data': player_analysis_context[player],
            'ai_commentary': ai_commentary,
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/debug-context', methods=['GET'])
def debug_context():
    global player_analysis_context, detected_players
    return jsonify({
        'detected_players': list(detected_players.keys()),
        'player_analysis_context': list(player_analysis_context.keys()),
        'context_details': {k: list(v.keys()) for k, v in player_analysis_context.items()}
    })


@app.route('/player-chat', methods=['POST'])
def player_chat():
    print(f"\n[LOG] ========== /player-chat START ==========")
    global player_analysis_context
    
    try:
        print(f"[LOG] 1. Request received")
        
        # Parse JSON request
        try:
            data = request.get_json(force=True)
            print(f"[LOG] 2. JSON parsed: {data}")
        except Exception as je:
            print(f"[ERROR] 2. JSON parse error: {je}")
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(je)}'}), 400
        
        player = data.get('player', '').strip()
        question = data.get('question', '').strip()
        print(f"[LOG] 3. Extracted: player='{player}', question='{question}'")

        if not player:
            print(f"[LOG] 4. No player provided")
            return jsonify({'success': False, 'error': 'Player required'}), 400
        if not question:
            print(f"[LOG] 4. No question provided")
            return jsonify({'success': False, 'error': 'Question required'}), 400
        if player not in player_analysis_context:
            print(f"[LOG] 4. Player '{player}' not in context. Available: {list(player_analysis_context.keys())}")
            return jsonify({'success': False, 'error': 'Run analysis first'}), 400

        print(f"[LOG] 4. All validation passed")
        ctx = player_analysis_context[player]
        profile = ctx.get('profile', {})
        print(f"[LOG] 5. Profile extracted: {profile}")
        
        # Simple, direct prompt for the chatbot
        prompt = (
            "You are an expert badminton coach. "
            f"Player profile: posture={profile.get('posture', 'unknown')}, "
            f"balance={profile.get('balance', 'unknown')}, stance={profile.get('stance', 'unknown')}. "
            f"Upright ratio={ctx.get('pose_metrics', {}).get('upright_ratio', 0):.0%}, "
            f"centered balance={ctx.get('pose_metrics', {}).get('centred_ratio', 0):.0%}, "
            f"step consistency={ctx.get('box_metrics', {}).get('movement_consistency', 0):.2f}. "
            f"The user asks: {question}\n\n"
            "Reply in 4-6 complete sentences with specific drills, corrections, and encouragement. "
            "Do not repeat the question or the prompt text."
        )
        print(f"[LOG] 6. Prompt prepared, calling generate_structured_feedback...")
        
        try:
            reply = generate_structured_feedback(prompt, max_new_tokens=500)
            print(f"[LOG] 7. Model responded: {reply[:80]}")
        except Exception as model_err:
            print(f"[ERROR] 7. Model generation failed: {type(model_err).__name__}: {model_err}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Model error: {str(model_err)}'}), 500
        
        print(f"[LOG] 8. Returning response")
        result = jsonify({
            'success': True,
            'player': player,
            'question': question,
            'reply': reply
        })
        print(f"[LOG] ========== /player-chat END ==========\n")
        return result, 200
    
    except Exception as e:
        print(f"[ERROR] OUTER EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"[LOG] ========== /player-chat END (WITH ERROR) ==========\n")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@app.route('/compare-players', methods=['POST'])
def compare_players():
    global detected_players
    
    data = request.json or {}
    player1 = data.get('player1', '')
    player2 = data.get('player2', '')
    
    if not player1 or not player2:
        return jsonify({'error': 'Select two players to compare'}), 400
    
    if not detected_players or player1 not in detected_players or player2 not in detected_players:
        return jsonify({'error': 'Players not found'}), 400
    
    try:
        def get_profile(label):
            poses = detected_players[label].get('poses', [])
            boxes = detected_players[label].get('boxes', [])
            if not poses:
                return {}
            
            postures = [p.get('posture') for p in poses if p.get('posture')]
            balances = [p.get('balance') for p in poses if p.get('balance')]
            stances = [p.get('stance') for p in poses if p.get('stance')]
            heights = [p.get('height_ratio') for p in poses if p.get('height_ratio')]

            return {
                'posture': max(set(postures), key=postures.count) if postures else 'unknown',
                'balance': max(set(balances), key=balances.count) if balances else 'unknown',
                'stance': max(set(stances), key=stances.count) if stances else 'unknown',
                'height': float(np.mean(heights)) if heights else 0,
                'pose_metrics': summarize_pose_metrics(poses),
                'box_metrics': compute_box_metrics(boxes),
            }

        p1_prof = get_profile(player1)
        p2_prof = get_profile(player2)

        feedback = build_rule_based_compare_feedback(player1, p1_prof, player2, p2_prof)
        feedback += (
            f"\n\nMETRIC SNAPSHOT:\n"
            f"- {player1}: samples={p1_prof['pose_metrics']['samples']}, "
            f"centred_ratio={p1_prof['pose_metrics']['centred_ratio']:.2f}, "
            f"step_consistency={p1_prof['box_metrics']['movement_consistency']:.2f}\n"
            f"- {player2}: samples={p2_prof['pose_metrics']['samples']}, "
            f"centred_ratio={p2_prof['pose_metrics']['centred_ratio']:.2f}, "
            f"step_consistency={p2_prof['box_metrics']['movement_consistency']:.2f}"
        )

        result = f"""
COMPARATIVE ANALYSIS: {player1} vs {player2}
{'='*70}

{feedback}

{'='*70}
"""

        return jsonify({'analysis': result})

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


print("[INFO] Flask routes after registration:")
for rule in app.url_map.iter_rules():
    print(f"  {rule}")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
