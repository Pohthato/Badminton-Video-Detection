"""Tracking step extracted from Untitled2.ipynb."""

from data_processing import process_video_and_track_objects


def run_tracking(video_path: str):
    """Run video tracking and return tracking artifacts."""
    return process_video_and_track_objects(video_path)
