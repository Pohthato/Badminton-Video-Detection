#!/usr/bin/env python3
"""Create a Colab deployment zip from the latest local app files."""

from pathlib import Path
import zipfile


INCLUDED_FILES = [
    "app.py",
    "data_processing.py",
    "llm_feedback.py",
    "notebook_feedback.py",
    "notebook_movement.py",
    "notebook_pipeline.py",
    "notebook_summary.py",
    "notebook_tracking.py",
    "player_analysis.py",
    "visualization_utils.py",
    "setup_hf_token.py",
    "colab_setup.py",
    "MODEL_INFO.md",
    "Badminton_Analyzer_GPU_Colab.ipynb",
    "yolov8n.pt",
    "templates/index.html",
    "templates/analysis.html",
    "templates/chat.html",
    "templates/base.html",
    "static/styles.css",
]


def create_gpu_deployment_zip():
    """Create a zip file with the current source tree for Colab."""
    base_dir = Path(__file__).resolve().parent
    zip_path = base_dir / "badminton_analyzer_gpu_deployment.zip"

    print("Creating GPU deployment package...")
    print(f"Output: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for relative_path in INCLUDED_FILES:
            full_path = base_dir / relative_path
            if full_path.exists():
                zipf.write(full_path, relative_path)
                print(f"[OK] Added: {relative_path}")
            else:
                print(f"[MISSING] {relative_path}")

    print(f"\nDeployment package created: {zip_path}")
    print("Upload this zip file to Google Colab.")
    print("Extract it into /content/badminton_analyzer and run the app there.")


if __name__ == "__main__":
    create_gpu_deployment_zip()
