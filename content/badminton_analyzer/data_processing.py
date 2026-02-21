import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

def process_video_and_track_objects(video_path):
    """
    Loads a video, performs object detection and tracking using YOLO,
    creates a DataFrame of tracked objects, and calculates movement metrics.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - df_tracking (pd.DataFrame): DataFrame with detailed tracking data.
            - fps (float): Frames per second of the video.
            - player_movement_summary (pd.DataFrame): Summary of player movement.
            - shuttlecock_movement_summary (pd.DataFrame): Summary of shuttlecock movement.
            - model (YOLO): The loaded YOLO model.
    """
    print(f"Processing video: {video_path}")

    # 1. Video Loading and Object Tracking
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None, None, None, None, None
    else:
        print(f"Video file '{video_path}' opened successfully.")

    # Load the pre-trained YOLO model
    model = YOLO('yolov8n.pt')

    # Define the target_classes list
    target_classes = ['person', 'sports ball']

    # Initialize an empty list called tracked_objects_data
    tracked_objects_data = []

    # Get the class IDs for 'person' and 'sports ball'
    target_class_ids = [
        k for k, v in model.names.items() if v in target_classes
    ]

    # Loop through each frame of the video
    frame_idx = 0
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Perform tracking on the current frame, limiting to target classes
        results = model.track(frame, persist=True, classes=target_class_ids, verbose=False)

        # Process results
        for r in results:
            if r.boxes.id is not None:
                for box, cls_id, conf, track_id in zip(
                    r.boxes.xyxy,
                    r.boxes.cls,
                    r.boxes.conf,
                    r.boxes.id
                ):
                    class_name = model.names[int(cls_id)]
                    tracked_objects_data.append({
                        'frame_id': frame_idx,
                        'track_id': int(track_id),
                        'class_name': class_name,
                        'class_id': int(cls_id),
                        'bbox': box.tolist(),
                        'confidence': float(conf)
                    })
        frame_idx += 1

    # Release the video_capture object
    video_capture.release()

    print(f"Tracking complete. Processed {frame_idx} frames and recorded {len(tracked_objects_data)} tracked objects.")

    # Convert the tracked_objects_data list into a pandas DataFrame
    df_tracking = pd.DataFrame(tracked_objects_data)

    # Calculate the center coordinates (x_center, y_center) for each bounding box
    df_tracking[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df_tracking['bbox'].tolist(), index=df_tracking.index)
    df_tracking['x_center'] = (df_tracking['x1'] + df_tracking['x2']) / 2
    df_tracking['y_center'] = (df_tracking['y1'] + df_tracking['y2']) / 2

    # Re-initialize video_capture again to get the frame rate.
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file at {video_path} to get FPS.")
        fps = 30
    else:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Warning: Video FPS reported as 0. Defaulting to 30 FPS.")
            fps = 30
        print(f"Video Frame Rate (FPS) retrieved: {fps}")
        video_capture.release()

    # Sort df_tracking to correctly calculate displacement
    df_tracking = df_tracking.sort_values(by=['track_id', 'class_name', 'frame_id']).reset_index(drop=True)

    # Calculate the displacement and speed
    df_tracking['prev_x_center'] = df_tracking.groupby(['track_id', 'class_name'])['x_center'].shift(1)
    df_tracking['prev_y_center'] = df_tracking.groupby(['track_id', 'class_name'])['y_center'].shift(1)
    df_tracking['distance_moved'] = np.sqrt(
        (df_tracking['x_center'] - df_tracking['prev_x_center'])**2 +
        (df_tracking['y_center'] - df_tracking['prev_y_center'])**2
    ).fillna(0)

    if fps > 0:
        df_tracking['speed'] = df_tracking['distance_moved'] * fps
    else:
        df_tracking['speed'] = 0

    # Create player_movement_summary DataFrame
    player_movement_summary = df_tracking[df_tracking['class_name'] == 'person'].groupby('track_id').agg(
        total_distance_moved=('distance_moved', 'sum'),
        average_speed=('speed', 'mean')
    ).reset_index()

    # Create shuttlecock_movement_summary DataFrame
    shuttlecock_movement_summary = df_tracking[df_tracking['class_name'] == 'sports ball'].groupby('track_id').agg(
        total_distance_moved=('distance_moved', 'sum'),
        average_speed=('speed', 'mean')
    ).reset_index()

    print("Initial data processing and tracking complete.")
    return df_tracking, fps, player_movement_summary, shuttlecock_movement_summary, model
