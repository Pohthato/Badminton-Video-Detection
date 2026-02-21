"""Primary player movement analysis extracted from Untitled2.ipynb."""

from player_analysis import analyze_primary_player_movement


def run_primary_player_analysis(df_tracking, fps: float, video_path: str):
    """Analyze the primary player with perspective adjustment."""
    return analyze_primary_player_movement(df_tracking, fps, video_path)
