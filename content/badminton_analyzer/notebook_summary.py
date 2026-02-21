"""Summary builder extracted from Untitled2.ipynb."""


def build_player_analysis_summary(
    primary_player_track_id: int,
    total_adjusted_distance: float,
    average_adjusted_speed: float,
) -> str:
    """Build a concise, prompt-friendly summary for coaching feedback."""
    return (
        f"Track ID: {int(primary_player_track_id)}\n"
        f"Total Adjusted Distance: {total_adjusted_distance:.2f} pixels\n"
        f"Average Adjusted Speed: {average_adjusted_speed:.2f} pixels/frame\n\n"
        "Trajectory: Player primarily moves in the lower-left court quadrant, "
        "occasionally shifting center. Movement is confined, shows repetitive "
        "returns to a home base, and is fluid.\n"
        "Activity Hotspots: Significant hotspot in lower-left frame "
        "(x=100-250, y=600-800 pixels). Less activity in top/right areas.\n"
        "Perspective Adjustment Impact: Movement estimates are weighted to reduce "
        "distance distortion from camera perspective."
    )
