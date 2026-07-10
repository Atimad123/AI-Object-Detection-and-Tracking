"""
=========================================================
metrics.py
---------------------------------------------------------
Metrics utilities for Object Detection Dashboard

Author : Atimad BEL CAID
=========================================================
"""

import time
import pandas as pd
from collections import Counter


class Metrics:
    """
    Compute detection statistics.
    """

    def __init__(self):

        self.previous_time = time.time()

        self.current_time = time.time()

        self.fps = 0

        self.inference_time = 0

    # =====================================================
    # FPS
    # =====================================================

    def update_fps(self):

        self.current_time = time.time()

        elapsed = self.current_time - self.previous_time

        if elapsed > 0:
            self.fps = 1 / elapsed

        self.previous_time = self.current_time

        return round(self.fps, 2)

    # =====================================================
    # INFERENCE TIME
    # =====================================================

    def update_inference(self, start_time):

        self.inference_time = (
            time.time() - start_time
        ) * 1000

        return round(self.inference_time, 2)

    # =====================================================
    # TOTAL OBJECTS
    # =====================================================

    @staticmethod
    def total_objects(detections):

        return len(detections)

    # =====================================================
    # COUNT PER CLASS
    # =====================================================

    @staticmethod
    def class_counter(detections):

        classes = [

            det["class_name"]

            for det in detections

        ]

        return dict(Counter(classes))

    # =====================================================
    # AVERAGE CONFIDENCE
    # =====================================================

    @staticmethod
    def average_confidence(detections):

        if len(detections) == 0:

            return 0

        conf = [

            det["confidence"]

            for det in detections

        ]

        avg = sum(conf) / len(conf)

        return round(avg * 100, 2)

    # =====================================================
    # DATAFRAME
    # =====================================================

    @staticmethod
    def dataframe(detections):

        rows = []

        for det in detections:

            rows.append({

                "ID": det["track_id"],

                "Class": det["class_name"],

                "Confidence":

                    f"{det['confidence']*100:.1f}%"

            })

        return pd.DataFrame(rows)

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self, detections):

        summary = {

            "fps": self.fps,

            "objects": self.total_objects(detections),

            "avg_confidence":

                self.average_confidence(detections),

            "classes":

                self.class_counter(detections)

        }

        return summary

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.previous_time = time.time()

        self.current_time = time.time()

        self.fps = 0

        self.inference_time = 0


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    metrics = Metrics()

    sample = [

        {

            "track_id": 1,

            "class_name": "person",

            "confidence": 0.98

        },

        {

            "track_id": 2,

            "class_name": "car",

            "confidence": 0.94

        },

        {

            "track_id": 3,

            "class_name": "person",

            "confidence": 0.91

        }

    ]

    print("=" * 60)

    print("Total Objects :",

          metrics.total_objects(sample))

    print()

    print("Average Confidence :",

          metrics.average_confidence(sample))

    print()

    print("Classes :",

          metrics.class_counter(sample))

    print()

    print(metrics.dataframe(sample))

    print("=" * 60)