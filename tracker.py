"""
tracker.py
---------------------------------------------------------
Object Tracking using YOLOv8 + ByteTrack

Features
---------
- Read video
- Object Detection
- Object Tracking
- Display FPS
- Save output video

Author:
Atimad BEL CAID
"""

import cv2
import time

from detector import ObjectDetector

from config import (
    INPUT_VIDEO,
    OUTPUT_VIDEO,
    WINDOW_NAME,
    SAVE_OUTPUT,
    SHOW_FPS
)


class ObjectTracker:
    """
    Video Object Tracking.
    """

    def __init__(self):

        self.detector = ObjectDetector()

    # =====================================================
    # Process Video
    # =====================================================

    def process_video(self):

        cap = cv2.VideoCapture(INPUT_VIDEO)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video:\n{INPUT_VIDEO}"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = None

        if SAVE_OUTPUT:

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                OUTPUT_VIDEO,
                fourcc,
                fps,
                (width, height)
            )

        previous_time = time.time()

        print("=" * 60)
        print("Processing video...")
        print("=" * 60)

        while True:

            success, frame = cap.read()

            if not success:
                break

            # =============================================
            # Detection + Tracking
            # =============================================

            annotated_frame = self.detector.draw(
                frame,
                tracking=True
            )

            # =============================================
            # FPS
            # =============================================

            if SHOW_FPS:

                current_time = time.time()

                fps_value = 1 / (current_time - previous_time)

                previous_time = current_time

                cv2.putText(

                    annotated_frame,

                    f"FPS : {fps_value:.2f}",

                    (20, 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 255, 255),

                    2

                )

            # =============================================
            # Show
            # =============================================

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame
            )

            # =============================================
            # Save
            # =============================================

            if SAVE_OUTPUT:

                writer.write(
                    annotated_frame
                )

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        print("=" * 60)
        print("Tracking completed.")
        print(f"Output saved to:\n{OUTPUT_VIDEO}")
        print("=" * 60)


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    tracker = ObjectTracker()

    tracker.process_video()