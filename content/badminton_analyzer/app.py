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

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"[WARNING] MediaPipe import failed: {e}")
    mp = None

app = Flask(__name__, static_folder='static', template_folder='templates')

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
            print("[INFO] Loading HF analyzer (FLAN-T5-large)...")
            tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
            model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large').to('cpu')
            hf_analyzer = {'tokenizer': tokenizer, 'model': model}
            print("[INFO] HF analyzer loaded successfully (CPU)")
        except Exception as e:
            print(f"[ERROR] Failed to load HF analyzer: {e}")
            import traceback
            traceback.print_exc()
            hf_analyzer = False


def generate_structured_feedback(prompt, max_new_tokens=260):
    """Generate cleaner, non-echoed coaching text with deterministic decoding."""
    tokenizer = hf_analyzer['tokenizer']
    model = hf_analyzer['model']

    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


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

    if posture == "upright":
        strengths.append("Maintains a tall torso that supports faster visual tracking and recovery.")
    else:
        weaknesses.append("Posture is too compressed, reducing reach and slowing directional change.")
        drills.append("Shadow split-step + recovery: 4x45s, focus on chest up and neutral spine.")

    if balance == "centred":
        strengths.append("Center of mass stays more controlled between shots.")
    else:
        weaknesses.append("Weight shifts off-center, causing late recovery after wide movements.")
        drills.append("Lateral lunge hold with racket reach: 3x8 each side, 2-second stabilize on landing.")

    if stance == "wide":
        strengths.append("Wide base provides stability on defensive contacts.")
        weaknesses.append("Base may be too wide during recovery, reducing first-step quickness.")
        drills.append("Split-step width drill: 5x30s, land shoulder-width then explode to corners.")
    elif stance == "narrow":
        weaknesses.append("Narrow base reduces stability under pressure.")
        drills.append("Mini-band squat walk + split-step: 3x40s to build stable receiving base.")
    else:
        strengths.append("Moderate base gives a workable balance of stability and mobility.")

    if height_ratio >= 2.0:
        strengths.append("Body alignment appears efficient for maintaining vertical control.")
    else:
        weaknesses.append("Low height ratio suggests excess forward fold in movement phases.")
        drills.append("Hip hinge wall taps + racket overhead reach: 3x10 controlled reps.")

    if len(drills) < 3:
        drills.append("6-point court footwork pattern: 4 rounds x 40s work / 20s rest, prioritize clean recovery.")
    if len(drills) < 3:
        drills.append("Two-shuttle reaction starts: 3x10 reps, commit to first step within 300 ms.")
    drills = drills[:3]

    improvements.extend([
        "Faster recovery to base should improve consistency in longer rallies.",
        "Better balance control will reduce unforced errors during directional changes.",
        "Cleaner posture mechanics will improve shot quality under fatigue.",
    ])

    if pose_metrics.get("balance_switch_rate", 0.0) > 0.45:
        weaknesses.append("Balance pattern is inconsistent across frames (high switch rate).")
        drills[0] = "Split-step freeze + recover: 4x30s, hold landing 1 second before first step."
    if pose_metrics.get("stance_switch_rate", 0.0) > 0.5:
        weaknesses.append("Base width changes too frequently, indicating inconsistent preparation.")
    if pose_metrics.get("vertical_mobility", 0.0) > 75:
        weaknesses.append("Excess vertical movement suggests wasted energy during transitions.")
    if box_metrics.get("movement_consistency", 0.0) < 0.45:
        weaknesses.append("Footwork tempo is inconsistent between steps.")
    if box_metrics.get("avg_step_px", 0.0) < 8:
        weaknesses.append("Average displacement per sampled frame is low, suggesting limited court coverage.")

    if not strengths:
        strengths.append(f"{player} shows consistent effort and repeatable movement habits that are coachable.")
    if not weaknesses:
        weaknesses.append("No major technical fault dominates; focus now should be efficiency and repeatability.")

    strengths.insert(0, f"Observed over {pose_metrics['samples']} sampled pose frames and {box_metrics['frames']} tracked boxes.")
    improvements.append(
        f"Targeting step consistency above 0.60 (current {box_metrics['movement_consistency']:.2f}) should improve rally control."
    )

    return (
        "STRENGTHS:\n- " + "\n- ".join(strengths[:3]) + "\n\n"
        "WEAKNESSES:\n- " + "\n- ".join(weaknesses[:4]) + "\n\n"
        "IMPROVEMENT DRILLS:\n- " + "\n- ".join(drills) + "\n\n"
        "EXPECTED IMPROVEMENTS:\n- " + "\n- ".join(improvements)
    )


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


@app.route("/")
def index():
    return render_template("index.html")


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route('/detect-humans', methods=['POST'])
def detect_humans():
    """
    Upload video, detect humans with YOLO across multiple frames,
    extract pose data with MediaPipe, and save for analysis.
    """
    global current_video_path, detected_players
    
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
        
        while frame_count < 150:  # First 150 frames (~5 sec at 30fps)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every Nth frame for analysis
            if frame_count % frame_sample_rate == 0:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                # YOLO detection
                results = yolo_model(frame, conf=0.5)
                detections = []
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append((x1, y1, x2, y2))

                if len(detections) > 10:
                    detections = detections[:10]

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
        
        # Keep strongest tracks and map to Player1..N
        ranked_tracks = sorted(
            [(tid, tdata) for tid, tdata in tracks.items() if len(tdata["boxes"]) >= 3],
            key=lambda item: len(item[1]["boxes"]),
            reverse=True,
        )
        detected_players = {}
        for idx, (tid, tdata) in enumerate(ranked_tracks[:10], start=1):
            detected_players[f"Player{idx}"] = {
                "track_id": tid,
                "boxes": tdata["boxes"],
                "poses": tdata["poses"],
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

        # Generate deterministic coaching text
        feedback = build_rule_based_player_feedback(
            player,
            dominant_posture,
            dominant_balance,
            dominant_stance,
            avg_height,
            pose_metrics,
            box_metrics,
        )

        # Final formatted output
        result = f"""
COACHING ANALYSIS: {player}
======================================================================

TECHNICAL PROFILE:
  Posture: {dominant_posture}
  Balance: {dominant_balance}
  Stance: {dominant_stance}
  Height Ratio: {avg_height:.2f}

FEEDBACK:
{feedback}

======================================================================
"""

        return jsonify({'analysis': result})

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



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


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
