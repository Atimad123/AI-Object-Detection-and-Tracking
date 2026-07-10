"""
=========================================================
detector.py
---------------------------------------------------------
YOLOv8 Object Detection & Tracking
=========================================================
"""

import cv2
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    TRACKER,
    PERSIST_TRACKS,
)


class ObjectDetector:
    """
    YOLOv8 Object Detector & Tracker
    """

    def __init__(self):

        print("=" * 60)
        print("Loading YOLOv8 model...")
        print("=" * 60)

        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names

        print("Model loaded successfully!")
        print("=" * 60)

    # =====================================================
    # IMAGE DETECTION
    # =====================================================

    def detect(self, image, confidence=None):

        if confidence is None:
            confidence = CONFIDENCE_THRESHOLD

        results = self.model.predict(
            image,
            conf=confidence,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        result = results[0]

        annotated = result.plot()

        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return annotated, detections

        for index, box in enumerate(result.boxes):

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            class_id = int(box.cls[0])

            detections.append(
                {
                    "track_id": index + 1,
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return annotated, detections

    # =====================================================
    # VIDEO / WEBCAM TRACKING
    # =====================================================

    def track(self, frame, confidence=None):

        if confidence is None:
            confidence = CONFIDENCE_THRESHOLD

        try:

            results = self.model.track(
                frame,
                conf=confidence,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                tracker=TRACKER,
                persist=PERSIST_TRACKS,
                verbose=False,
            )

        except Exception:

            # Fallback si ByteTrack échoue
            results = self.model.predict(
                frame,
                conf=confidence,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )

        result = results[0]

        annotated = result.plot()

        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return annotated, detections

        ids = None

        if hasattr(result.boxes, "id") and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()

        for index, box in enumerate(result.boxes):

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            class_id = int(box.cls[0])

            conf = round(float(box.conf[0]), 3)

            if ids is not None and index < len(ids):
                track_id = int(ids[index])
            else:
                track_id = index + 1

            detections.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return annotated, detections
    # =====================================================
    # DETECT FROM IMAGE FILE
    # =====================================================

    def detect_file(self, image_path):
        """
        Detect objects from an image file.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        return self.detect(image)

    # =====================================================
    # DETECT FROM VIDEO FILE
    # =====================================================

    def detect_video(self, video_path):
        """
        Generator that yields annotated frames and detections.
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        while True:

            success, frame = cap.read()

            if not success:
                break

            annotated, detections = self.track(frame)

            yield annotated, detections

        cap.release()

    # =====================================================
    # DRAW DETECTIONS
    # =====================================================

    def draw(self, image, detections):
        """
        Draw detections manually.
        """

        output = image.copy()

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            color = (0, 255, 0)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = (
                f'ID:{det["track_id"]} '
                f'{det["class_name"]} '
                f'{det["confidence"]:.2f}'
            )

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        return output

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    def info(self):
        """
        Return model information.
        """

        return {
            "Model": str(MODEL_PATH),
            "Classes": len(self.class_names),
            "Image Size": IMAGE_SIZE,
            "Confidence Threshold": CONFIDENCE_THRESHOLD,
            "IOU Threshold": IOU_THRESHOLD,
            "Tracker": TRACKER,
            "Persistent Tracking": PERSIST_TRACKS,
        }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    detector = ObjectDetector()

    print("\nModel Information\n")

    for key, value in detector.info().items():
        print(f"{key}: {value}")