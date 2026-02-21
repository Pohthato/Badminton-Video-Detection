"""CLI pipeline assembled from Untitled2.ipynb modules."""

import argparse

from notebook_feedback import GPT2Coach
from notebook_movement import run_primary_player_analysis
from notebook_summary import build_player_analysis_summary
from notebook_tracking import run_tracking


def run_pipeline(video_path: str, gemini_api_key: str | None = None):
    tracking = run_tracking(video_path)
    df_tracking, fps, player_summary, shuttle_summary, _ = tracking

    if df_tracking is None or df_tracking.empty:
        raise RuntimeError("No tracking data produced from video.")

    (
        primary_player_track_id,
        df_primary_player,
        primary_player_adjusted_summary,
        total_adjusted_distance,
        average_adjusted_speed,
    ) = run_primary_player_analysis(df_tracking, fps, video_path)

    summary = build_player_analysis_summary(
        primary_player_track_id,
        total_adjusted_distance,
        average_adjusted_speed,
    )

    # Optional Gemini path
    gemini_feedback = None
    if gemini_api_key:
        from llm_feedback import generate_coaching_feedback

        gemini_feedback = generate_coaching_feedback(summary, gemini_api_key)

    # Local GPT-2 path
    local_coach = GPT2Coach()
    hf_feedback = local_coach.generate_feedback(summary)

    return {
        "primary_player_track_id": int(primary_player_track_id),
        "player_movement_summary": player_summary,
        "shuttlecock_movement_summary": shuttle_summary,
        "primary_player_adjusted_summary": primary_player_adjusted_summary,
        "df_primary_player": df_primary_player,
        "analysis_summary": summary,
        "hf_feedback": hf_feedback,
        "gpt2_feedback": hf_feedback,
        "gemini_feedback": gemini_feedback,
    }


def main():
    parser = argparse.ArgumentParser(description="Run badminton notebook pipeline as Python modules.")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--gemini-api-key", default=None, help="Optional Gemini API key")
    args = parser.parse_args()

    result = run_pipeline(args.video_path, args.gemini_api_key)
    print("\n--- Analysis Summary ---")
    print(result["analysis_summary"])
    print("\n--- Hugging Face Feedback ---")
    print(result["hf_feedback"])
    if result["gemini_feedback"]:
        print("\n--- Gemini Feedback ---")
        print(result["gemini_feedback"])


if __name__ == "__main__":
    main()
