"""
=========================================================
helpers.py
---------------------------------------------------------
Utility functions for Object Detection & Tracking

Author : Atimad BEL CAID
=========================================================
"""

import cv2
import os
import tempfile
import numpy as np


# ==========================================================
# IMAGE FUNCTIONS
# ==========================================================

def load_image(uploaded_file):
    """
    Convert a Streamlit uploaded image to OpenCV format.

    Parameters
    ----------
    uploaded_file

    Returns
    -------
    image (numpy.ndarray)
    """

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    return image


# ==========================================================
# SAVE TEMP FILE
# ==========================================================

def save_uploaded_file(uploaded_file):
    """
    Save uploaded image/video temporarily.

    Returns
    -------
    file_path
    """

    suffix = os.path.splitext(uploaded_file.name)[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    return temp_file.name


# ==========================================================
# VIDEO FUNCTIONS
# ==========================================================

def open_video(video_path):
    """
    Open a video.

    Returns
    -------
    cv2.VideoCapture
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video : {video_path}"
        )

    return cap


def get_video_info(cap):
    """
    Return video information.
    """

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    return {

        "width": width,

        "height": height,

        "fps": fps,

        "frames": total_frames

    }


# ==========================================================
# VIDEO WRITER
# ==========================================================

def create_video_writer(
        output_path,
        width,
        height,
        fps
):
    """
    Create output video writer.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(

        output_path,

        fourcc,

        fps,

        (width, height)

    )

    return writer


# ==========================================================
# IMAGE CONVERSION
# ==========================================================

def bgr_to_rgb(image):
    """
    Convert OpenCV BGR image to RGB.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# DRAW TEXT
# ==========================================================

def draw_text(
        image,
        text,
        x,
        y,
        color=(0, 255, 0)
):
    """
    Draw text on image.
    """

    cv2.putText(

        image,

        text,

        (x, y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        color,

        2

    )


# ==========================================================
# DRAW FPS
# ==========================================================

def draw_fps(
        image,
        fps
):
    """
    Draw FPS.
    """

    draw_text(

        image,

        f"FPS : {fps:.2f}",

        20,

        30,

        (0, 255, 255)

    )


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

def create_directory(path):
    """
    Create directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


# ==========================================================
# FILE SIZE
# ==========================================================

def file_size(path):
    """
    Return file size in MB.
    """

    size = os.path.getsize(path)

    return round(
        size / (1024 * 1024),
        2
    )


# ==========================================================
# FORMAT TIME
# ==========================================================

def format_time(seconds):
    """
    Convert seconds to hh:mm:ss
    """

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    secs = int(seconds % 60)

    return f"{hours:02}:{minutes:02}:{secs:02}"


# ==========================================================
# RANDOM COLOR
# ==========================================================

def random_color(class_id):
    """
    Generate a deterministic color for each class.
    """

    np.random.seed(class_id)

    color = np.random.randint(
        0,
        255,
        size=3
    )

    return (
        int(color[0]),
        int(color[1]),
        int(color[2])
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 50)

    print("Helpers module loaded successfully.")

    print("=" * 50)