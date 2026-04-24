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
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Suppress HF token warning and other warnings
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import warnings
warnings.filterwarnings('ignore')

# MediaPipe imports
try:
    import mediapipe as mp
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
pose_model = None
pose_model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hf_analyzer = None
mp_pose = None
hog_detector = None
current_video_path = None  # Store video path for analysis
detected_players = {}  # Store player bounding boxes: {label: [(x1,y1,x2,y2), ...frames]}
player_analysis_context = {}  # Keep last player analytic state for chatbot follow-up
last_detected_frame = None  # Base64 preview image from the latest detection

COCO_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}


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



def init_pose_model():
    global pose_model, pose_model_device
    if pose_model is None:
        try:
            print("[INFO] Loading skeleton pose model...")
            pose_model = keypointrcnn_resnet50_fpn(pretrained=True, progress=False, pretrained_backbone=True)
            pose_model.eval()
            pose_model.to(pose_model_device)
            print(f"[INFO] Skeleton pose model loaded on {pose_model_device}")
        except Exception as e:
            print(f"[WARNING] Failed to load skeleton pose model: {e}")
            pose_model = False
    return pose_model


def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / (boxAArea + boxBArea - interArea)


def safe_angle(point_a, point_b, point_c):
    """Return the angle ABC in degrees, or None when geometry is unreliable."""
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return None
    cosine = np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def score_is_reliable(scores, indices, threshold=0.35):
    return all(float(scores[idx]) >= threshold for idx in indices)


SKELETON_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11),
    (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def draw_skeleton(image, keypoints, scores, score_thresh=0.4, color=(0, 255, 0)):
    for start, end in SKELETON_EDGES:
        if scores[start] >= score_thresh and scores[end] >= score_thresh:
            p1 = tuple(map(int, keypoints[start]))
            p2 = tuple(map(int, keypoints[end]))
            cv2.line(image, p1, p2, color, 2, lineType=cv2.LINE_AA)
    for idx, kp in enumerate(keypoints):
        if scores[idx] >= score_thresh:
            point = tuple(map(int, kp))
            cv2.circle(image, point, 3, color, -1, lineType=cv2.LINE_AA)


def run_skeleton_detector(frame):
    init_pose_model()
    if not pose_model or pose_model is False:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(pose_model_device)
    with torch.no_grad():
        outputs = pose_model([tensor])

    if not outputs:
        return None

    output = outputs[0]
    return {
        'boxes': output['boxes'].cpu().numpy(),
        'scores': output['scores'].cpu().numpy(),
        'keypoints': output['keypoints'].cpu().numpy(),
    }


def init_hog_detector():
    global hog_detector
    if hog_detector is None:
        hog_detector = cv2.HOGDescriptor()
        hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog_detector


def detect_people_with_hog(frame):
    hog = init_hog_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.03, finalThreshold=2.0)
    boxes = []
    for x, y, w, h in rects:
        if w < 30 or h < 50:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def merge_additional_detections(primary_boxes, new_boxes, min_iou=0.55):
    result = list(primary_boxes)
    for box in new_boxes:
        if not any(box_iou(box, existing) > min_iou for existing in result):
            result.append(box)
    return result


def compute_pose_features_from_keypoints(keypoints, box, frame_shape):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    if keypoints.shape[0] < 17:
        return None

    key_xy = keypoints[:, :2]
    key_conf = keypoints[:, 2]

    left_shoulder = key_xy[COCO_KEYPOINTS['left_shoulder']]
    right_shoulder = key_xy[COCO_KEYPOINTS['right_shoulder']]
    left_hip = key_xy[COCO_KEYPOINTS['left_hip']]
    right_hip = key_xy[COCO_KEYPOINTS['right_hip']]
    left_knee = key_xy[COCO_KEYPOINTS['left_knee']]
    right_knee = key_xy[COCO_KEYPOINTS['right_knee']]
    left_ankle = key_xy[COCO_KEYPOINTS['left_ankle']]
    right_ankle = key_xy[COCO_KEYPOINTS['right_ankle']]
    nose = key_xy[COCO_KEYPOINTS['nose']]

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    spine_vec = shoulder_center - hip_center

    vertical_angle = abs(np.degrees(np.arctan2(spine_vec[0], spine_vec[1]))) if np.linalg.norm(spine_vec) > 1e-6 else 90.0
    is_upright = vertical_angle < 25.0
    box_center_x = (x1 + x2) / 2.0
    width = frame_shape[1]
    balance = 'centred' if abs(box_center_x - width / 2.0) < width * 0.12 else 'off_centre'

    ankle_distance = np.linalg.norm(left_ankle - right_ankle)
    hip_distance = np.linalg.norm(left_hip - right_hip)
    stance_width = float(ankle_distance if ankle_distance > 1.0 else hip_distance)
    stance_ratio = stance_width / max(box_width, 1.0)
    if stance_ratio > 0.45:
        stance = 'wide'
    elif stance_ratio < 0.25:
        stance = 'narrow'
    else:
        stance = 'moderate'

    shoulder_tilt_deg = abs(float(np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1], right_shoulder[0] - left_shoulder[0]))))
    hip_tilt_deg = abs(float(np.degrees(np.arctan2(right_hip[1] - left_hip[1], right_hip[0] - left_hip[0]))))

    left_knee_angle = safe_angle(left_hip, left_knee, left_ankle) if score_is_reliable(key_conf, [COCO_KEYPOINTS['left_hip'], COCO_KEYPOINTS['left_knee'], COCO_KEYPOINTS['left_ankle']]) else None
    right_knee_angle = safe_angle(right_hip, right_knee, right_ankle) if score_is_reliable(key_conf, [COCO_KEYPOINTS['right_hip'], COCO_KEYPOINTS['right_knee'], COCO_KEYPOINTS['right_ankle']]) else None
    knee_angles = [angle for angle in [left_knee_angle, right_knee_angle] if angle is not None]
    avg_knee_flexion_deg = float(np.mean(knee_angles)) if knee_angles else None

    head_offset_ratio = None
    if score_is_reliable(key_conf, [COCO_KEYPOINTS['nose'], COCO_KEYPOINTS['left_hip'], COCO_KEYPOINTS['right_hip']]):
        head_offset_ratio = float(abs(nose[0] - hip_center[0]) / max(box_width, 1.0))

    low_ready = False
    if avg_knee_flexion_deg is not None and avg_knee_flexion_deg < 155.0 and vertical_angle < 38.0:
        low_ready = True

    asymmetry_flag = shoulder_tilt_deg > 14.0 or hip_tilt_deg > 14.0
    posture_label = 'upright' if is_upright else 'bent/crouched'
    if low_ready and vertical_angle < 32.0:
        posture_label = 'athletic_ready'

    return {
        'posture': posture_label,
        'balance': balance,
        'stance_width': stance_width,
        'height_ratio': float(box_height / max(box_width, 1.0)),
        'position_y': float(y1),
        'stance': stance,
        'trunk_lean_deg': float(vertical_angle),
        'shoulder_tilt_deg': shoulder_tilt_deg,
        'hip_tilt_deg': hip_tilt_deg,
        'avg_knee_flexion_deg': avg_knee_flexion_deg,
        'head_offset_ratio': head_offset_ratio,
        'low_ready': low_ready,
        'asymmetry_flag': asymmetry_flag,
        'skeleton_confidence': float(np.mean(key_conf)),
        'keypoints': key_xy.tolist(),
        'keypoint_scores': key_conf.tolist(),
    }


def extract_pose_data(image, box, pose_results=None):
    """Extract pose information from actual skeleton keypoints or fallback to box shape."""
    if pose_results is not None:
        best_index = find_best_pose_match(box, pose_results, min_score=0.3, min_iou=0.1)
        if best_index is not None:
            keypoints = pose_results['keypoints'][best_index]
            pose_data = compute_pose_features_from_keypoints(keypoints, box, image.shape)
            if pose_data is not None:
                return pose_data

    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    box_height = y2 - y1
    box_width = x2 - x1
    aspect_ratio = box_height / (box_width + 1e-6)
    is_upright = aspect_ratio > 1.5
    box_center_x = (x1 + x2) / 2
    return {
        'posture': 'upright' if is_upright else 'bent/crouched',
        'balance': 'centred' if (box_center_x > w * 0.3 and box_center_x < w * 0.7) else 'off_centre',
        'stance_width': float(box_width),
        'height_ratio': float(aspect_ratio),
        'position_y': float(y1),
        'stance': 'narrow' if box_width < 60 else 'wide' if box_width > 100 else 'moderate',
        'trunk_lean_deg': 0.0 if is_upright else 35.0,
        'shoulder_tilt_deg': None,
        'hip_tilt_deg': None,
        'avg_knee_flexion_deg': None,
        'head_offset_ratio': None,
        'low_ready': False,
        'asymmetry_flag': False,
        'skeleton_confidence': 0.0,
    }


def init_hf_analyzer():
    global hf_analyzer
    if hf_analyzer is None:
        preferred_model = os.getenv('BADMINTON_AI_MODEL', 'google/flan-t5-base').strip() or 'google/flan-t5-base'
        candidate_models = [preferred_model]
        if preferred_model != 'google/flan-t5-small':
            candidate_models.append('google/flan-t5-small')

        for model_name in candidate_models:
            try:
                print(f"[INFO] Loading HF analyzer ({model_name})...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model.cpu()
                hf_analyzer = {'tokenizer': tokenizer, 'model': model, 'model_name': model_name}
                print(f"[INFO] HF analyzer loaded successfully: {model_name}")
                return
            except Exception as e:
                print(f"[WARNING] Failed to load analyzer model {model_name}: {e}")

        print("[WARNING] HF analyzer not available, using rule-based feedback")
        hf_analyzer = False


def is_valid_coaching_feedback(text, min_length=15):
    """Validate that generated text is coherent coaching feedback and not garbled."""
    if not text or len(text) < min_length:
        return False
    
    # Strong garbled detection - stricter criteria
    garbled_signs = [
        text.count('\n') > 15,  # Excessive line breaks
        text.count('  ') > 10,  # Multiple consecutive spaces (fixing issues)
        any(word in text.lower() for word in [
            'arrogance', 'rewordability', 'sportsbooks', 'wz of weight', 'yr at current', 
            'eakin', 'tutti', 'hone', 'listen on', 'cheat their coach', 'highest speed did both',
            'bet upon', 'the pace by his power', 'good for each guy', 'www m-stam', 'httpwww'
        ]),
        text.lower().count('pro') > 10,  # Excessive 'pro' 
        len(set(text.split())) / len(text.split()) < 0.2 if text.split() else False,  # High repetition
        any(len(word) > 50 for word in text.split()),  # Extremely long garbled words
        'http' in text.lower() or 'www' in text.lower(),  # URLs in output (garbled)
        text.count('?') > 4 and text.count('.') < text.count('?'),  # More questions than statements
        '????' in text or '!!!!' in text,  # Repeated punctuation
    ]
    
    if any(garbled_signs):
        return False
    
    # Must have coherent structure - look for real coaching content
    has_coaching_content = (
        any(keyword in text.lower() for keyword in ['posture', 'balance', 'movement', 'court', 'footwork', 'technique']) or
        'improve' in text.lower() or 'practice' in text.lower() or 'drill' in text.lower() or 'strength' in text.lower()
    )
    
    return has_coaching_content


def format_percent(value):
    return f"{value * 100:.0f}%"


def format_optional_metric(value, suffix="", digits=1):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def build_ai_coaching_prompt(player, question, profile, pose_metrics, box_metrics, feedback_data, summary_report):
    strengths = feedback_data.get('strengths', [])
    weaknesses = feedback_data.get('weaknesses', [])
    drills = feedback_data.get('drills', [])

    prompt = f"""
You are a badminton coach writing a short performance review from skeleton-analysis metrics.
Stay grounded in the provided measurements. Do not invent shots, scores, rallies, handedness, or match context.
Prioritize skeleton observations over generic movement commentary.
Write a long, structured coaching response with these section titles:
Technical Snapshot
What Looks Good
What Needs Work
Specific Drills
Next Step For This Player
Each section should contain 2-4 complete sentences or bullet points.
Use plain coaching language. Be specific and practical.

Player: {player}
Optional user question: {question or 'none'}
Dominant profile: posture={profile.get('posture', 'unknown')}, balance={profile.get('balance', 'unknown')}, stance={profile.get('stance', 'unknown')}

Skeleton metrics:
- upright ratio: {format_percent(pose_metrics.get('upright_ratio', 0.0))}
- athletic ready ratio: {format_percent(pose_metrics.get('athletic_ready_ratio', 0.0))}
- centred balance ratio: {format_percent(pose_metrics.get('centred_ratio', 0.0))}
- wide stance ratio: {format_percent(pose_metrics.get('wide_ratio', 0.0))}
- avg trunk lean: {format_optional_metric(pose_metrics.get('avg_trunk_lean_deg'), ' deg')}
- avg shoulder tilt: {format_optional_metric(pose_metrics.get('avg_shoulder_tilt_deg'), ' deg')}
- avg hip tilt: {format_optional_metric(pose_metrics.get('avg_hip_tilt_deg'), ' deg')}
- avg knee flexion: {format_optional_metric(pose_metrics.get('avg_knee_flexion_deg'), ' deg')}
- low ready ratio: {format_percent(pose_metrics.get('low_ready_ratio', 0.0))}
- asymmetry ratio: {format_percent(pose_metrics.get('asymmetry_ratio', 0.0))}
- skeleton confidence: {pose_metrics.get('avg_skeleton_confidence', 0.0):.2f}

Movement metrics:
- tracked frames: {box_metrics.get('frames', 0)}
- avg step size: {box_metrics.get('avg_step_px', 0.0):.1f}px
- movement consistency: {box_metrics.get('movement_consistency', 0.0):.2f}
- total movement: {box_metrics.get('total_movement_px', 0.0):.1f}px

Rule-based strengths:
{chr(10).join(f"- {item}" for item in strengths[:3]) if strengths else "- none"}

Rule-based weaknesses:
{chr(10).join(f"- {item}" for item in weaknesses[:3]) if weaknesses else "- none"}

Recommended drills:
{chr(10).join(f"- {item}" for item in drills[:3]) if drills else "- none"}

Ground-truth summary:
{summary_report}
""".strip()
    return prompt


def build_pose_focus_commentary(player, profile, pose_metrics, box_metrics, feedback_data, question=""):
    posture = profile.get('posture', 'unknown')
    balance = profile.get('balance', 'unknown')
    stance = profile.get('stance', 'unknown')
    upright_ratio = pose_metrics.get('upright_ratio', 0.0)
    athletic_ready_ratio = pose_metrics.get('athletic_ready_ratio', 0.0)
    low_ready_ratio = pose_metrics.get('low_ready_ratio', 0.0)
    asymmetry_ratio = pose_metrics.get('asymmetry_ratio', 0.0)
    trunk_lean = pose_metrics.get('avg_trunk_lean_deg')
    shoulder_tilt = pose_metrics.get('avg_shoulder_tilt_deg')
    hip_tilt = pose_metrics.get('avg_hip_tilt_deg')
    knee_flexion = pose_metrics.get('avg_knee_flexion_deg')
    avg_step = box_metrics.get('avg_step_px', 0.0)
    movement_consistency = box_metrics.get('movement_consistency', 0.0)

    technical_snapshot = [
        f"{player}'s tracked skeleton profile currently reads as {posture} posture, {balance} balance, and a {stance} stance.",
        f"The pose sample shows {format_percent(upright_ratio)} upright frames and {format_percent(athletic_ready_ratio)} frames in an athletic ready position.",
        f"Average trunk lean is {format_optional_metric(trunk_lean, ' deg')}, shoulder tilt is {format_optional_metric(shoulder_tilt, ' deg')}, and hip tilt is {format_optional_metric(hip_tilt, ' deg')}.",
    ]

    issue_parts = []
    if asymmetry_ratio >= 0.35:
        issue_parts.append(
            f"Upper- and lower-body alignment is inconsistent, with asymmetry appearing in {format_percent(asymmetry_ratio)} of the pose frames"
        )
    if shoulder_tilt is not None and shoulder_tilt > 10.0:
        issue_parts.append(f"the shoulders are tilting by about {shoulder_tilt:.1f} degrees")
    if hip_tilt is not None and hip_tilt > 10.0:
        issue_parts.append(f"the hips are tilting by about {hip_tilt:.1f} degrees")
    if knee_flexion is not None and knee_flexion > 160.0:
        issue_parts.append(f"the legs are often too straight at roughly {knee_flexion:.1f} degrees")
    if low_ready_ratio < 0.35:
        issue_parts.append("the player is not getting into a low ready base often enough")
    if movement_consistency < 0.55:
        issue_parts.append(f"movement timing is only moderately repeatable at {movement_consistency:.2f}")

    if not issue_parts:
        issue_parts.append("the main next step is making the current skeleton shape more repeatable from frame to frame")

    strengths = []
    if upright_ratio >= 0.55:
        strengths.append("The upper body is staying reasonably organized through a good portion of the tracked frames.")
    if athletic_ready_ratio >= 0.30 or low_ready_ratio >= 0.40:
        strengths.append("The player is reaching a usable ready position often enough to build on.")
    if movement_consistency >= 0.6:
        strengths.append(f"Movement timing is fairly repeatable at {movement_consistency:.2f}, which is a useful foundation for footwork work.")
    if avg_step >= 12:
        strengths.append(f"Average step size is {avg_step:.1f}px, so there is already some court coverage to work with.")
    if not strengths:
        strengths.append("The clip still gives a useful baseline, even if the current technique is not yet stable.")

    drills = feedback_data.get('drills', [])
    improvement = feedback_data.get('improvements', ['Better pose repeatability should translate into cleaner footwork.'])[0]
    drill_1 = drills[0] if drills else "Shadow footwork with a chest-up posture check every third movement."
    drill_2 = drills[1] if len(drills) > 1 else "Split-step to lunge recovery work with a pause in the ready position."
    weakness_summary = (
        "The biggest technical issue from the skeleton data is that "
        + ", ".join(issue_parts[:3])
        + ". This matters because unstable posture and base shape make it harder to push explosively into the next shot and recover cleanly to center."
    )

    question_lower = question.lower()
    question_focus = "Use the skeleton findings as the main reference point for the next practice block."
    if "smash" in question_lower:
        question_focus = (
            "For the smash specifically, the body line has to stay tall enough for full shoulder rotation while the legs remain loaded underneath. "
            "If the chest collapses or the base gets too narrow, power leaks before contact."
        )
    elif "footwork" in question_lower or "movement" in question_lower:
        question_focus = (
            "For footwork, the goal is not just moving more. The goal is reaching the shuttle from a balanced base and recovering without losing trunk shape."
        )
    elif "posture" in question_lower or "balance" in question_lower:
        question_focus = (
            "For posture and balance, the main win is keeping the chest organized above the hips instead of letting the body tip and twist through each plant."
        )

    sections = [
        "Technical Snapshot\n"
        + "\n".join(f"- {item}" for item in technical_snapshot),
        "What Looks Good\n"
        + "\n".join(f"- {item}" for item in strengths[:4]),
        "What Needs Work\n"
        + "\n".join([
            f"- {weakness_summary}",
            f"- The player is currently averaging {avg_step:.1f}px per step with movement consistency at {movement_consistency:.2f}, so the target is cleaner shape without losing coverage.",
        ]),
        "Specific Drills\n"
        + "\n".join([
            f"- {drill_1}",
            f"- {drill_2}",
            "- Film the next drill block from the same camera angle so the posture changes can be compared directly.",
        ]),
        "Next Step For This Player\n"
        + "\n".join([
            f"- {question_focus}",
            f"- Expected improvement if this is cleaned up: {improvement}",
        ]),
    ]

    return "\n\n".join(sections)


def generate_structured_feedback(prompt, max_new_tokens=300):
    """Generate coaching feedback with a constrained, metrics-aware prompt."""
    try:
        if hf_analyzer is None:
            init_hf_analyzer()

        if not hf_analyzer or hf_analyzer is False:
            return None  # Let caller handle fallback

        tokenizer = hf_analyzer['tokenizer']
        model = hf_analyzer['model']
        inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        lowered = result.lower()
        for leak in [
            "you are a badminton coach",
            "write a long, structured coaching response",
            "skeleton metrics:",
            "movement metrics:",
            "rule-based strengths:",
            "recommended drills:",
            "ground-truth summary:",
        ]:
            if leak in lowered:
                return None

        return result if result else None
        
    except Exception as e:
        print(f"[ERROR] AI generation failed: {e}")
        return None


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
            "athletic_ready_ratio": 0.0,
            "low_ready_ratio": 0.0,
            "asymmetry_ratio": 0.0,
            "avg_trunk_lean_deg": None,
            "avg_shoulder_tilt_deg": None,
            "avg_hip_tilt_deg": None,
            "avg_knee_flexion_deg": None,
            "avg_head_offset_ratio": None,
            "avg_skeleton_confidence": 0.0,
            "skeleton_frame_ratio": 0.0,
        }

    postures = [p.get("posture", "unknown") for p in valid]
    balances = [p.get("balance", "unknown") for p in valid]
    stances = [p.get("stance", "unknown") for p in valid]
    heights = np.array([float(p.get("height_ratio", 0.0)) for p in valid], dtype=float)
    widths = np.array([float(p.get("stance_width", 0.0)) for p in valid], dtype=float)
    y_positions = np.array([float(p.get("position_y", 0.0)) for p in valid], dtype=float)
    trunk_leans = [float(p.get("trunk_lean_deg")) for p in valid if p.get("trunk_lean_deg") is not None]
    shoulder_tilts = [float(p.get("shoulder_tilt_deg")) for p in valid if p.get("shoulder_tilt_deg") is not None]
    hip_tilts = [float(p.get("hip_tilt_deg")) for p in valid if p.get("hip_tilt_deg") is not None]
    knee_flexions = [float(p.get("avg_knee_flexion_deg")) for p in valid if p.get("avg_knee_flexion_deg") is not None]
    head_offsets = [float(p.get("head_offset_ratio")) for p in valid if p.get("head_offset_ratio") is not None]
    skeleton_confidences = [float(p.get("skeleton_confidence", 0.0)) for p in valid]
    skeleton_supported = [p for p in valid if float(p.get("skeleton_confidence", 0.0)) > 0.05]

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
        "athletic_ready_ratio": postures.count("athletic_ready") / n,
        "low_ready_ratio": sum(1 for p in valid if p.get("low_ready")) / n,
        "asymmetry_ratio": sum(1 for p in valid if p.get("asymmetry_flag")) / n,
        "avg_trunk_lean_deg": float(np.mean(trunk_leans)) if trunk_leans else None,
        "avg_shoulder_tilt_deg": float(np.mean(shoulder_tilts)) if shoulder_tilts else None,
        "avg_hip_tilt_deg": float(np.mean(hip_tilts)) if hip_tilts else None,
        "avg_knee_flexion_deg": float(np.mean(knee_flexions)) if knee_flexions else None,
        "avg_head_offset_ratio": float(np.mean(head_offsets)) if head_offsets else None,
        "avg_skeleton_confidence": float(np.mean(skeleton_confidences)) if skeleton_confidences else 0.0,
        "skeleton_frame_ratio": len(skeleton_supported) / n,
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
    """Deterministic coaching text with skeleton metrics taking priority."""
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
    athletic_ready_ratio = pose_metrics.get("athletic_ready_ratio", 0.0)
    low_ready_ratio = pose_metrics.get("low_ready_ratio", 0.0)
    asymmetry_ratio = pose_metrics.get("asymmetry_ratio", 0.0)
    trunk_lean = pose_metrics.get("avg_trunk_lean_deg")
    shoulder_tilt = pose_metrics.get("avg_shoulder_tilt_deg")
    hip_tilt = pose_metrics.get("avg_hip_tilt_deg")
    knee_flexion = pose_metrics.get("avg_knee_flexion_deg")
    skeleton_confidence = pose_metrics.get("avg_skeleton_confidence", 0.0)
    skeleton_frame_ratio = pose_metrics.get("skeleton_frame_ratio", 0.0)
    movement_consistency = box_metrics.get("movement_consistency", 0.0)
    avg_step = box_metrics.get("avg_step_px", 0.0)
    samples = pose_metrics.get("samples", 0)
    frames = box_metrics.get("frames", 0)

    # START with frame/sample count - always relevant
    strengths.append(
        f"Observed {samples} pose frames and {frames} tracking samples, with skeleton support on {format_percent(skeleton_frame_ratio)} of usable frames."
    )
    if skeleton_confidence >= 0.45:
        strengths.append(f"Skeleton confidence is solid at {skeleton_confidence:.2f}, so the posture read is reasonably trustworthy.")

    # SKELETON-DRIVEN POSTURE ANALYSIS
    if athletic_ready_ratio >= 0.35 or low_ready_ratio >= 0.45:
        strengths.append("Frequently reaches an athletic ready shape, which should help push-off speed and recovery.")
    elif knee_flexion is not None and knee_flexion > 160.0:
        weaknesses.append("The skeleton shows a tall base with limited knee bend, so the ready position is too high for quick first steps.")
        drills.append("Split-step to hold drill: 4x20s, pause in a lower loaded base before each push-off.")

    if upright_ratio >= 0.6:
        strengths.append("Maintains a stable trunk through many frames, supporting sustained court presence.")
    elif upright_ratio >= 0.4:
        weaknesses.append("Posture control is mixed; the trunk angle changes too often instead of staying stable through recovery.")
        drills.append("Shadow rally with chest-up checkpoints every third movement for 4x40s.")
    else:
        weaknesses.append("Body alignment is consistently collapsed forward; focus on a taller chest while keeping the legs loaded.")
        drills.append("Wall-assisted posture resets: 3x15s, reset to neutral spine between movements.")

    if trunk_lean is not None and trunk_lean > 28.0:
        weaknesses.append(f"Average trunk lean is {trunk_lean:.1f} degrees, which suggests the upper body is tipping too far instead of stacking over the base.")
    elif trunk_lean is not None and trunk_lean < 18.0:
        strengths.append(f"Average trunk lean stays controlled at {trunk_lean:.1f} degrees.")

    # BALANCE ANALYSIS - meaningful thresholds
    if centred_ratio >= 0.65:
        strengths.append("Excellent center balance preservation across movement transitions.")
    elif centred_ratio >= 0.45:
        weaknesses.append("Center balance is decent but not yet reliable under change of direction.")
    elif centred_ratio < 0.35:
        weaknesses.append("Center balance is inconsistent; practice weighted recovery to base position.")
        drills.append("Single-leg balance holds: 3x10s each leg, then side-to-side recovery.")

    if asymmetry_ratio >= 0.35:
        weaknesses.append("Shoulder and hip levels are uneven too often, which can reduce movement efficiency and hitting stability.")
        drills.append("Mirror shadow footwork: 3x30s, keep shoulders and hips level through each plant.")
    elif asymmetry_ratio <= 0.18 and shoulder_tilt is not None and hip_tilt is not None:
        strengths.append("Shoulder and hip alignment stay relatively level during most tracked poses.")

    if shoulder_tilt is not None and shoulder_tilt > 12.0:
        weaknesses.append(f"Shoulder tilt averages {shoulder_tilt:.1f} degrees, so upper-body control is leaking during movement.")
    if hip_tilt is not None and hip_tilt > 12.0:
        weaknesses.append(f"Hip tilt averages {hip_tilt:.1f} degrees, which points to unstable loading on directional pushes.")

    # STANCE WIDTH ANALYSIS - only flag when problematic
    if wide_ratio >= 0.7:
        strengths.append("Consistently wide stance provides strong base for defensive movement.")
    elif wide_ratio <= 0.2:
        weaknesses.append("Narrow stance limits lateral stability; widen base under pressure situations.")
        drills.append("Band-assisted squat walks: 3x30s, maintain stance width through deceleration.")
    # Otherwise skip - moderate stance is fine

    # HEIGHT RATIO - only if problematic
    if height_ratio < 1.3 and posture == "bent/crouched":
        weaknesses.append("Low height ratio combined with poor posture; prioritize chest-up mechanics.")
        drills.append("Mirror posture drill: 3x8 reps, reset to full extension between each movement.")

    # BALANCE AND STANCE SWITCHING - only flag if excessive (>0.4)
    if balance_switch_rate > 0.5:
        weaknesses.append("Frequent balance adjustments suggest unstable weight transfer.")
        drills.append("Split-step hold + recovery: 4x30s, stabilize footwork before directional commit.")
    
    if stance_switch_rate > 0.6:
        weaknesses.append("High stance variability may indicate insufficient lower-body control.")
        if "Mini-band squat" not in " ".join(drills):
            drills.append("Mini-band squat walks: 3x40s, locked foot placement drill.")

    # VERTICAL MOBILITY - only flag extreme cases
    if vertical_mobility > 100:
        weaknesses.append("Excessive vertical bounce wastes energy; lower center-of-gravity for efficiency.")

    # MOVEMENT QUALITY METRICS - focus on actionable signals
    if movement_consistency > 0.65 and avg_step > 15:
        strengths.append(f"Strong movement consistency ({movement_consistency:.2f}) with good court coverage ({avg_step:.1f}px avg step).")
    elif movement_consistency < 0.4 and avg_step > 12:
        weaknesses.append("Steps are variable; standardize recovery footwork for predictability.")
        if "6-point court" not in " ".join(drills):
            drills.append("6-point court pattern: 3x40s work / 20s rest, locked recovery routes.")
    elif avg_step < 8:
        weaknesses.append("Small average step size limits court coverage per movement; increase stride length in practice.")

    # Build improvements based on what was flagged
    if weaknesses:
        if any("knee" in w.lower() or "ready" in w.lower() or "trunk" in w.lower() for w in weaknesses):
            improvements.append("A cleaner ready position should improve first-step explosiveness without sacrificing stability.")
        if any("balance" in w.lower() for w in weaknesses):
            improvements.append("Improved weight transfer will reduce errors on direction changes.")
        if any("stance" in w.lower() or "posture" in w.lower() or "vertical" in w.lower() or "shoulder" in w.lower() or "hip" in w.lower() for w in weaknesses):
            improvements.append("Better postural stability extends rally-duration endurance.")
        if any("step" in w.lower() or "movement" in w.lower() for w in weaknesses):
            improvements.append("More consistent footwork patterns accelerate first-step response time.")
    
    # Add drills only if we identified specific weaknesses
    if not drills:
        drills.append("Shadow footwork with mirror: 4x45s, focus on smooth transitions between bases.")
    elif len(drills) < 2:
        drills.append("Two-shuttle reaction drill: 3x10 reps, maximize acceleration from static ready.")

    # Ensure we have some improvements
    if not improvements:
        improvements.append("Consolidate current technique with consistent drill work for match reliability.")

    # Limit lists to reasonable sizes
    return {
        'strengths': strengths[:5],
        'weaknesses': weaknesses[:4],
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


def build_player_analysis_report(player, dominant_posture, dominant_balance, dominant_stance, avg_height, pose_metrics, box_metrics, feedback_data):
    """Create a long personalized coaching summary from metrics and feedback."""
    frames = box_metrics.get('frames', 0)
    movement_consistency = box_metrics.get('movement_consistency', 0.0)
    avg_step = box_metrics.get('avg_step_px', 0.0)
    upright_ratio = pose_metrics.get('upright_ratio', 0.0)
    athletic_ready_ratio = pose_metrics.get('athletic_ready_ratio', 0.0)
    centred_ratio = pose_metrics.get('centred_ratio', 0.0)
    balance_switch_rate = pose_metrics.get('balance_switch_rate', 0.0)
    stance_switch_rate = pose_metrics.get('stance_switch_rate', 0.0)
    vertical_mobility = pose_metrics.get('vertical_mobility', 0.0)
    asymmetry_ratio = pose_metrics.get('asymmetry_ratio', 0.0)
    trunk_lean = pose_metrics.get('avg_trunk_lean_deg')
    shoulder_tilt = pose_metrics.get('avg_shoulder_tilt_deg')
    hip_tilt = pose_metrics.get('avg_hip_tilt_deg')
    knee_flexion = pose_metrics.get('avg_knee_flexion_deg')

    strengths = feedback_data.get('strengths', [])
    weaknesses = feedback_data.get('weaknesses', [])
    drills = feedback_data.get('drills', [])
    improvements = feedback_data.get('improvements', [])

    summary = []
    summary.append(
        f"Over {frames} tracked movement frames, {player} displayed a {dominant_posture} posture pattern with {dominant_balance} balance and a {dominant_stance} stance. "
        f"The height ratio is {avg_height:.2f}, which is consistent with current movement style."
    )

    if athletic_ready_ratio >= 0.35:
        summary.append(
            f"The skeleton data shows an athletic ready position in {athletic_ready_ratio:.2f} of frames, which is a positive sign for push-off readiness."
        )
    elif knee_flexion is not None:
        summary.append(
            f"The average knee angle is {knee_flexion:.1f} degrees, suggesting the ready base could be loaded more consistently."
        )

    if upright_ratio >= 0.6:
        summary.append(
            f"Upright posture is a relative strength ({upright_ratio:.2f} of frames upright), which supports higher court coverage and reduces fatigue."
        )
    elif upright_ratio >= 0.35:
        summary.append(
            f"Posture is mixed ({upright_ratio:.2f} upright ratio). This means the player can maintain good mechanics some of the time but needs more consistency in keeping the chest up."
        )
    else:
        summary.append(
            f"Posture is a key weakness: only {upright_ratio:.2f} of frames are upright. The player should focus on raising the body line during rallies."
        )

    if trunk_lean is not None:
        summary.append(
            f"Average trunk lean is {trunk_lean:.1f} degrees, which helps quantify how much the body is tipping forward through movement."
        )
    if shoulder_tilt is not None or hip_tilt is not None:
        summary.append(
            f"Alignment is being tracked through shoulder tilt ({format_optional_metric(shoulder_tilt, ' deg')}) and hip tilt ({format_optional_metric(hip_tilt, ' deg')}). "
            f"Asymmetry appears in {asymmetry_ratio:.2f} of frames."
        )

    if centred_ratio >= 0.6:
        summary.append(
            f"Center balance is solid ({centred_ratio:.2f}), indicating good recovery to the midline after shots."
        )
    elif centred_ratio >= 0.4:
        summary.append(
            f"Center balance is moderate ({centred_ratio:.2f}). The player should tighten recovery patterns to avoid being pulled off-center."
        )
    else:
        summary.append(
            f"Center balance is unstable ({centred_ratio:.2f}). Working on quicker returns to base and weight distribution will help reduce errors on wide rallies."
        )

    if movement_consistency >= 0.65:
        summary.append(
            f"Movement consistency is a strength ({movement_consistency:.2f}), which suggests the footwork pattern is repeatable across several rally positions."
        )
    else:
        summary.append(
            f"Movement consistency is lower than desired ({movement_consistency:.2f}). The player should practice drills that force repeatable step timing and recovery."
        )

    if avg_step >= 18:
        summary.append(
            f"Average step length is strong at {avg_step:.1f}px, supporting good coverage of the court."
        )
    elif avg_step >= 12:
        summary.append(
            f"Average step length is acceptable ({avg_step:.1f}px) but could be improved for faster court coverage."
        )
    else:
        summary.append(
            f"Average step size is small ({avg_step:.1f}px). Increasing step length while maintaining balance will improve reach and recovery."
        )

    if balance_switch_rate > 0.5 or stance_switch_rate > 0.6:
        summary.append(
            f"The skeleton and tracking signals show frequent base changes, with balance switching at {balance_switch_rate:.2f} and stance switching at {stance_switch_rate:.2f}. "
            "That usually means the player is reacting after landing instead of moving from a stable base."
        )

    if vertical_mobility > 100:
        summary.append(
            f"Vertical movement varies by {vertical_mobility:.1f}px, which may indicate extra bounce rather than efficient glide steps."
        )

    if weaknesses:
        summary.append("The top areas to address are:")
        for item in weaknesses[:3]:
            summary.append(f"- {item}")
    if drills:
        summary.append("Recommended practice focus:")
        for item in drills[:3]:
            summary.append(f"- {item}")

    if improvements:
        summary.append("If these adjustments are made, the expected improvement is:")
        for item in improvements[:2]:
            summary.append(f"- {item}")

    summary.append(
        "Keep the training specific: focus on one technical correction per session and use the drills above to build consistent movement patterns. "
        "This will create a stronger foundation before adding power or aggressive shot-making."
    )

    return "\n".join(summary)


def build_rule_based_chat_reply(player, question):
    ctx = player_analysis_context.get(player, {})
    profile = ctx.get('profile', {})
    pose_metrics = ctx.get('pose_metrics', {})
    box_metrics = ctx.get('box_metrics', {})

    upright_ratio = pose_metrics.get('upright_ratio', 0.0)
    centred_ratio = pose_metrics.get('centred_ratio', 0.0)
    movement_consistency = box_metrics.get('movement_consistency', 0.0)
    avg_step = box_metrics.get('avg_step_px', 0.0)

    question_lower = question.lower()
    if 'smash' in question_lower:
        return (
            "For a stronger smash, start with a higher contact point and a straighter arm action. "
            "Keep the non-racket shoulder down and rotate your hips through the shot. "
            "Practice shadow smashes with a focus on a full wrist snap at the end of the swing. "
            "If your current footwork is slow, do 6-point recovery drills immediately after each smash to rebuild court position."
        )
    if 'footwork' in question_lower or 'movement' in question_lower:
        return (
            f"Your footwork profile shows {movement_consistency:.2f} movement consistency and an average step of {avg_step:.1f}px. "
            "Focus on split-step timing, then move into the next position with a strong push from the outside foot. "
            "Use ladder drills and 6-point court coverage work to make the pattern more reliable under pressure."
        )
    if 'balance' in question_lower or 'posture' in question_lower:
        return (
            f"The player has {profile.get('balance', 'off-centre')} balance and {profile.get('posture', 'bent/crouched')} posture. "
            "In rally work, keep the chest up and drive the center of mass back to the middle after each shot. "
            "A split-step recovery drill with a posture check on every third shot will help this become automatic."
        )
    return (
        "The strongest improvement route is to prioritize one technical habit at a time. "
        "Start sessions with movement drills, then layer in posture and balance work. "
        "Keep repetitions deliberate: quality of execution matters more than quantity. "
        "This approach builds a predictable foundation before adding power or aggression."
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


def find_best_pose_match(box, pose_results, min_score=0.2, min_iou=0.1):
    """Find the pose detection that best overlaps a tracked person box."""
    if pose_results is None:
        return None

    best_index = None
    best_iou = 0.0
    for i, det_box in enumerate(pose_results.get('boxes', [])):
        score = float(pose_results['scores'][i])
        if score < min_score:
            continue
        det_box_tuple = tuple(map(float, det_box))
        current_iou = box_iou(box, det_box_tuple)
        if current_iou > best_iou:
            best_iou = current_iou
            best_index = i

    if best_index is None or best_iou < min_iou:
        return None
    return best_index


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
    Stage 1: Upload video and detect all people in the first frame ONLY.
    User will then select which player to analyze, triggering Stage 2 tracking.
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
        # Stage 1: Detect ALL people in FIRST FRAME ONLY - No skeleton tracking yet
        cap = cv2.VideoCapture(str(save_path))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Failed to read video'}), 400
        
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        print("[INFO] Stage 1: Detecting all people in first frame...")
        
        # YOLO detection with aggressive settings to catch EVERYONE
        results = yolo_model(frame, conf=0.15, verbose=False)  # Very low confidence to catch everyone
        detections = []
        min_box_width = max(30, int(w * 0.02))  # Extremely lenient
        min_box_height = max(40, int(h * 0.04))  # Extremely lenient
        max_box_width = int(w * 0.95)
        max_box_height = int(h * 0.99)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            if cls == 0 and conf >= 0.15:  # Detect any person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_width = x2 - x1
                box_height = y2 - y1
                area = box_width * box_height

                # Allow lots of detections
                if box_width < min_box_width or box_height < min_box_height:
                    continue
                if box_width > max_box_width or box_height > max_box_height:
                    continue

                aspect_ratio = box_height / (box_width + 1e-6)
                # Flexible aspect ratios
                if aspect_ratio < 0.4 or aspect_ratio > 6.0:
                    continue

                if area < (min_box_width * min_box_height * 0.25) or area > 400000:
                    continue

                detections.append((x1, y1, x2, y2))

        # HOG fallback for any missed people
        try:
            hog_boxes = detect_people_with_hog(frame)
            detections = merge_additional_detections(detections, hog_boxes, min_iou=0.4)
        except Exception as hog_err:
            print(f"[WARNING] HOG fallback failed: {hog_err}")

        # Pose model fallback for any remaining undetected persons
        try:
            pose_results = run_skeleton_detector(frame)
            if pose_results is not None:
                for i, score in enumerate(pose_results['scores']):
                    if score < 0.2:
                        continue
                    det_box = tuple(map(int, pose_results['boxes'][i]))
                    if det_box[2] <= det_box[0] or det_box[3] <= det_box[1]:
                        continue
                    if any(box_iou(det_box, existing) > 0.55 for existing in detections):
                        continue
                    box_width = det_box[2] - det_box[0]
                    box_height = det_box[3] - det_box[1]
                    if box_width < min_box_width or box_height < min_box_height:
                        continue
                    detections.append(det_box)
        except Exception as pose_err:
            print(f"[WARNING] pose fallback failed: {pose_err}")

        # NMS to remove overlaps - but keep all distinct people
        detections = apply_nms(detections, overlap_threshold=0.35)  # Very lenient NMS
        detections = filter_close_detections(detections, min_distance_px=25)  # Allow very close detections

        # Remove outliers but keep everyone else
        if len(detections) > 15:
            # Keep the 15 most confident detections
            scored = [(box, (box[2]-box[0]) * (box[3]-box[1])) for box in detections]
            scored.sort(key=lambda x: x[1], reverse=True)
            detections = [box for box, _ in scored[:15]]

        print(f"[INFO] Found {len(detections)} people in first frame")

        # Store initial detections (no poses yet)
        detected_players = {}
        for idx, (x1, y1, x2, y2) in enumerate(detections, start=1):
            detected_players[f"Player{idx}"] = {
                "initial_box": (x1, y1, x2, y2),
                "boxes": [],  # Will be filled during tracking
                "poses": [],  # Will be filled during tracking
                "metrics": {},
            }

        # Draw first frame with detections and full skeleton overlays when available.
        frame_with_boxes = frame.copy()
        pose_results = None
        try:
            pose_results = run_skeleton_detector(frame)
        except Exception as pose_draw_err:
            print(f"[WARNING] preview pose drawing failed: {pose_draw_err}")

        for idx, (x1, y1, x2, y2) in enumerate(detections, start=1):
            label = f"Player{idx}"
            pose_index = find_best_pose_match((x1, y1, x2, y2), pose_results, min_score=0.2, min_iou=0.1)
            if pose_index is not None:
                draw_skeleton(
                    frame_with_boxes,
                    pose_results['keypoints'][pose_index][:, :2],
                    pose_results['keypoints'][pose_index][:, 2],
                    score_thresh=0.25,
                    color=(0, 220, 255),
                )
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, label, (x1, max(y1 - 10, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        last_detected_frame = frame_b64

        return jsonify({
            'frame': frame_b64,
            'labels': list(detected_players.keys()),
            'count': len(detected_players),
            'message': f'Detected {len(detected_players)} people. Select one to analyze!'
        })
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


def track_selected_player():
    """
    Stage 2: After user selects a player from Stage 1, track that SPECIFIC player
    across the entire video with skeleton extraction and pose analysis.
    This creates much more data (hundreds of frames) for better analysis.
    """
    global current_video_path, detected_players
    
    data = request.json or {}
    player_label = data.get('player', '')
    
    if not player_label or player_label not in detected_players:
        return jsonify({'error': 'Invalid player selection'}), 400
    
    if not current_video_path:
        return jsonify({'error': 'No video loaded'}), 400
    
    try:
        print(f"[INFO] Stage 2: Tracking {player_label} across entire video with skeleton...")
        
        # Get the initial bounding box for this player
        initial_box = detected_players[player_label].get('initial_box')
        if not initial_box:
            return jsonify({'error': 'Player detection data missing'}), 400
        
        init_pose_model()
        
        # Track the selected player across the video
        cap = cv2.VideoCapture(current_video_path)
        frame_count = 0
        frame_sample_rate = 2  # Sample every 2nd frame for smooth tracking (lots of data)
        boxes = []  # Track boxes for this player only
        poses = []  # Track poses for this player only
        max_frames = 500  # Analyze up to 500 frames (much more than before)
        
        init_yolo()
        
        track_box = initial_box
        last_box = initial_box
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_sample_rate == 0:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                # Detect this player with YOLO
                results = yolo_model(frame, conf=0.15, verbose=False)
                detections = []
                
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    if cls == 0 and conf >= 0.15:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        # Very lenient constraints
                        if box_width < 25 or box_height < 45:
                            continue
                        if box_width > int(w * 0.98) or box_height > int(h * 0.99):
                            continue
                        
                        detections.append((x1, y1, x2, y2))

                if not detections:
                    try:
                        hog_boxes = detect_people_with_hog(frame)
                        detections = merge_additional_detections(detections, hog_boxes, min_iou=0.3)
                    except Exception as hog_err:
                        print(f"[WARNING] HOG stage-2 fallback failed: {hog_err}")

                if not detections:
                    try:
                        pose_results = run_skeleton_detector(frame)
                        if pose_results is not None:
                            for i, score in enumerate(pose_results['scores']):
                                if score < 0.2:
                                    continue
                                det_box = tuple(map(int, pose_results['boxes'][i]))
                                if det_box[2] <= det_box[0] or det_box[3] <= det_box[1]:
                                    continue
                                detections.append(det_box)
                    except Exception as pose_err:
                        print(f"[WARNING] stage-2 pose fallback failed: {pose_err}")

                best_match = None
                best_score = -999.0
                reference_box = track_box or last_box or initial_box
                reference_area = max(1.0, (reference_box[2] - reference_box[0]) * (reference_box[3] - reference_box[1]))
                for det_box in detections:
                    det_area = max(1.0, (det_box[2] - det_box[0]) * (det_box[3] - det_box[1]))
                    area_ratio = det_area / reference_area
                    if area_ratio < 0.25 or area_ratio > 4.0:
                        continue
                    iou = box_iou(reference_box, det_box)
                    center_dist = center_distance(reference_box, det_box)
                    score = iou * 100.0 - (center_dist * 0.02)
                    if score > best_score:
                        best_score = score
                        best_match = det_box

                if best_match is not None and best_score > 0.2:
                    track_box = best_match
                elif last_box is not None:
                    padded = (
                        max(0, last_box[0] - 8),
                        max(0, last_box[1] - 8),
                        min(w - 1, last_box[2] + 8),
                        min(h - 1, last_box[3] + 8),
                    )
                    track_box = padded
                else:
                    track_box = initial_box

                boxes.append(track_box)
                last_box = track_box
                
                pose_results = run_skeleton_detector(frame)
                pose = extract_pose_data(frame, track_box, pose_results)
                if pose is not None:
                    poses.append(pose)
                else:
                    poses.append({})  # Empty pose if extraction fails
            
            frame_count += 1
        
        cap.release()
        
        # Store the tracking data
        if boxes:
            metrics = compute_box_metrics(boxes)
            detected_players[player_label]['boxes'] = boxes
            detected_players[player_label]['poses'] = poses
            detected_players[player_label]['metrics'] = metrics
            
            print(f"[INFO] Tracked {player_label} for {len(boxes)} frames with {len([p for p in poses if p])} skeleton poses")
            
            return jsonify({
                'success': True,
                'player': player_label,
                'frames_tracked': len(boxes),
                'frames_with_skeleton': len([p for p in poses if p]),
                'message': f'Tracked {len(boxes)} frames! Now analyzing...'
            })
        else:
            return jsonify({'error': 'Failed to track player'}), 400
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Tracking failed: {str(e)}'}), 500


@app.route('/track-selected-player', methods=['POST'])
def track_selected_player_route():
    """Flask route wrapper for track_selected_player function"""
    return track_selected_player()


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
            feedback_data = {
                'strengths': [
                    'Player detected and tracked successfully.',
                    'The tracking appears stable even with limited motion.'
                ],
                'weaknesses': [
                    'Limited movement detected in this sample.',
                    'The current clip does not show enough court coverage.'
                ],
                'drills': [
                    'Practice more dynamic footwork and movement drills before re-recording.',
                    'Include more rally-style motion so the system can deliver deeper feedback.'
                ],
                'improvements': [
                    'Use active gameplay for richer analysis.',
                    'Capture motion across at least 10 seconds of court movement.'
                ],
                'metrics': {
                    'total_movement_px': total_movement,
                    'avg_step_px': avg_step,
                    'frames': box_metrics.get('frames', 0),
                }
            }

            feedback_text = (
                "STRENGTHS:\n- " + "\n- ".join(feedback_data['strengths']) + "\n\n"
                "WEAKNESSES:\n- " + "\n- ".join(feedback_data['weaknesses']) + "\n\n"
                "IMPROVEMENT DRILLS:\n- " + "\n- ".join(feedback_data['drills']) + "\n\n"
                "EXPECTED IMPROVEMENTS:\n- " + "\n- ".join(feedback_data['improvements'])
            )
            summary_report = (
                "This sample contains limited movement, so the coaching insights are conservative. "
                "The player was tracked successfully but the clip does not show enough court coverage or rally movement to fully assess dynamic footwork. "
                "For a deeper analysis, upload a video with more continuous rally play and directional changes."
            )
            ai_commentary = summary_report
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

            focus_issues = []
            if pose_metrics.get('centred_ratio', 0) < 0.5:
                focus_issues.append("center balance and weight control")
            if pose_metrics.get('upright_ratio', 0) < 0.5:
                focus_issues.append("vertical posture")
            if box_metrics.get('movement_consistency', 0) < 0.6:
                focus_issues.append("footwork consistency")
            if box_metrics.get('avg_step_px', 0) < 15:
                focus_issues.append("court coverage")
            focus_str = ", ".join(focus_issues[:3]) if focus_issues else "overall technique"

            summary_report = build_player_analysis_report(
                player,
                dominant_posture,
                dominant_balance,
                dominant_stance,
                avg_height,
                pose_metrics,
                box_metrics,
                feedback_data,
            )

            # Generate AI commentary from the actual skeleton metrics
            init_hf_analyzer()
            ai_commentary = build_pose_focus_commentary(
                player,
                {
                    'posture': dominant_posture,
                    'balance': dominant_balance,
                    'stance': dominant_stance,
                    'height_ratio': avg_height,
                },
                pose_metrics,
                box_metrics,
                feedback_data,
            )
            if hf_analyzer and hf_analyzer is not False:
                simple_ai_prompt = build_ai_coaching_prompt(
                    player,
                    f"Focus on {focus_str}.",
                    {
                        'posture': dominant_posture,
                        'balance': dominant_balance,
                        'stance': dominant_stance,
                        'height_ratio': avg_height,
                    },
                    pose_metrics,
                    box_metrics,
                    feedback_data,
                    summary_report,
                )
                generated_commentary = generate_structured_feedback(simple_ai_prompt, max_new_tokens=220)
                if generated_commentary and is_valid_coaching_feedback(generated_commentary):
                    ai_commentary = generated_commentary

            if not ai_commentary or not is_valid_coaching_feedback(ai_commentary):
                ai_commentary = build_pose_focus_commentary(
                    player,
                    {
                        'posture': dominant_posture,
                        'balance': dominant_balance,
                        'stance': dominant_stance,
                        'height_ratio': avg_height,
                    },
                    pose_metrics,
                    box_metrics,
                    feedback_data,
                )

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
            'ai_model': hf_analyzer.get('model_name') if hf_analyzer and hf_analyzer is not False else 'rule_based_pose_summary',
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

DETAILED COACHING SUMMARY:
{summary_report}

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
        pose_metrics = ctx.get('pose_metrics', {})
        box_metrics = ctx.get('box_metrics', {})
        feedback_data = ctx.get('feedback', {})
        print(f"[LOG] 5. Profile extracted: {profile}")
        
        prompt = build_ai_coaching_prompt(
            player,
            question,
            profile,
            pose_metrics,
            box_metrics,
            feedback_data,
            ctx.get('ai_commentary', '') or ctx.get('summary', ''),
        ) + "\n\nAnswer the user question directly while staying consistent with the measurements above."
        print(f"[LOG] 6. Prompt prepared, calling generate_structured_feedback...")
        
        try:
            reply = generate_structured_feedback(prompt, max_new_tokens=500)
            print(f"[LOG] 7. Model responded: {reply[:80] if reply else 'None'}")
        except Exception as model_err:
            print(f"[ERROR] 7. Model generation failed: {type(model_err).__name__}: {model_err}")
            import traceback
            traceback.print_exc()
            reply = None

        if not reply or not is_valid_coaching_feedback(reply):
            print("[LOG] 8. Using deterministic fallback chat response")
            reply = build_pose_focus_commentary(player, profile, pose_metrics, box_metrics, feedback_data, question=question)
            if question:
                reply = reply + "\n\nAdditional Answer To Your Question\n- " + build_rule_based_chat_reply(player, question)

        print(f"[LOG] 9. Returning response")
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
