"""
=========================================================
config.py
---------------------------------------------------------
Configuration file for AI Object Detection & Tracking

Author : Atimad BEL CAID
=========================================================
"""

import os
from ultralytics.utils.downloads import attempt_download_asset

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSETS_DIR = os.path.join(BASE_DIR, "assets")

IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")

OUTPUT_DIR = os.path.join(ASSETS_DIR, "output")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================================
# MODEL
# ==========================================================

# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "yolov8n.pt"

MODEL_PATH = os.path.join(
    MODELS_DIR,
    MODEL_NAME
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Please place yolov8n.pt inside the models folder."
    )

# ==========================================================
# YOLO PARAMETERS
# ==========================================================

CONFIDENCE_THRESHOLD = 0.40

IOU_THRESHOLD = 0.45

IMAGE_SIZE = 640

# ==========================================================
# TRACKER
# ==========================================================

TRACKER = "bytetrack.yaml"

PERSIST_TRACKS = True

# ==========================================================
# INPUT / OUTPUT FILES
# ==========================================================

INPUT_IMAGE = os.path.join(
    IMAGES_DIR,
    "street.jpg"
)

INPUT_VIDEO = os.path.join(
    VIDEOS_DIR,
    "traffic.mp4"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "result.mp4"
)

# ==========================================================
# STREAMLIT
# ==========================================================

PAGE_TITLE = "AI Object Detection & Tracking"

PAGE_ICON = "🎯"

LAYOUT = "wide"

# ==========================================================
# DISPLAY
# ==========================================================

WINDOW_NAME = "YOLOv8 Object Detection"

SHOW_FPS = True

SAVE_OUTPUT = True

# ==========================================================
# DASHBOARD COLORS
# ==========================================================

PRIMARY_COLOR = "#3B82F6"

SUCCESS_COLOR = "#22C55E"

WARNING_COLOR = "#F59E0B"

DANGER_COLOR = "#EF4444"

BACKGROUND_COLOR = "#0F172A"

CARD_COLOR = "#1E293B"

TEXT_COLOR = "#FFFFFF"

# ==========================================================
# SUPPORTED FILES
# ==========================================================

IMAGE_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png"
]

VIDEO_EXTENSIONS = [
    "mp4",
    "avi",
    "mov",
    "mkv"
]

# ==========================================================
# APPLICATION INFO
# ==========================================================

APP_NAME = "AI Object Detection Dashboard"

VERSION = "1.0.0"

AUTHOR = "Atimad BEL CAID"

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print(APP_NAME)
    print("=" * 60)

    print("Base Directory :", BASE_DIR)
    print("Images :", IMAGES_DIR)
    print("Videos :", VIDEOS_DIR)
    print("Output :", OUTPUT_DIR)
    print("Model :", MODEL_PATH)

    print("=" * 60)