"""
=========================================================
Webcam Detection
---------------------------------------------------------
Real-Time Object Detection with YOLOv8 + Streamlit WebRTC
=========================================================
"""

import time
import threading
import av
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

from detector import ObjectDetector
from utils.metrics import Metrics
from utils.charts import DashboardCharts

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Webcam Detection",
    page_icon="📷",
    layout="wide"
)

st.title("📷 Real-Time Webcam Detection")

st.markdown("""
Detect objects in real time using **YOLOv8 + ByteTrack**.
""")

st.divider()

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_detector():
    return ObjectDetector()

detector = load_detector()

charts = DashboardCharts()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙ Webcam Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.40,
    0.05,
)

show_boxes = st.sidebar.checkbox(
    "Show Bounding Boxes",
    True,
)

# ==========================================================
# SHARED DATA
# ==========================================================

class SharedState:

    def __init__(self):

        self.lock = threading.Lock()

        self.fps = 0

        self.objects = 0

        self.confidence = 0

        self.inference = 0

        self.class_counts = {}

        self.detections = []

        self.last_frame = None


shared = SharedState()

# ==========================================================
# RTC CONFIG
# ==========================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)

# ==========================================================
# VIDEO PROCESSOR
# ==========================================================

class YOLOProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = detector

        self.metrics = Metrics()

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        start = time.time()

        annotated, detections = self.detector.track(
            img,
            confidence,
        )

        inference = (time.time() - start) * 1000

        fps = self.metrics.update_fps()

        total = self.metrics.total_objects(detections)

        avg_conf = self.metrics.average_confidence(detections)

        class_counts = self.metrics.class_counter(detections)

        with shared.lock:

            shared.fps = fps

            shared.objects = total

            shared.confidence = avg_conf

            shared.inference = inference

            shared.class_counts = class_counts

            shared.detections = detections

            shared.last_frame = annotated.copy()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )
# ==========================================================
# START WEBCAM
# ==========================================================

st.subheader("🎥 Live Camera")

ctx = webrtc_streamer(
    key="yolo-webcam",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.info("▶ Cliquez sur START pour lancer la webcam.")

# ==========================================================
# LIVE DASHBOARD
# ==========================================================

import pandas as pd

metric_placeholder = st.empty()
table_placeholder = st.empty()
chart_placeholder = st.empty()

if "fps_history" not in st.session_state:
    st.session_state.fps_history = []

# ==========================================================
# READ SHARED DATA
# ==========================================================

with shared.lock:

    fps = shared.fps
    objects = shared.objects
    confidence_value = shared.confidence
    inference = shared.inference
    class_counts = shared.class_counts.copy()
    detections = shared.detections.copy()

# ==========================================================
# FPS HISTORY
# ==========================================================

st.session_state.fps_history.append(fps)

if len(st.session_state.fps_history) > 100:
    st.session_state.fps_history.pop(0)

fps_history = st.session_state.fps_history

# ==========================================================
# METRICS
# ==========================================================

with metric_placeholder.container():

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🎯 Objects",
        objects,
    )

    c2.metric(
        "⚡ FPS",
        f"{fps:.2f}",
    )

    c3.metric(
        "🧠 Confidence",
        f"{confidence_value:.2f}%",
    )

    c4.metric(
        "⏱ Inference",
        f"{inference:.2f} ms",
    )

# ==========================================================
# DETECTION TABLE
# ==========================================================

with table_placeholder.container():

    st.subheader("📋 Current Detections")

    if len(detections):

        df = pd.DataFrame(detections)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info("No object detected.")
# ==========================================================
# CHARTS
# ==========================================================

with chart_placeholder.container():

    if len(class_counts):

        left, right = st.columns(2)

        # --------------------------------------------------
        # BAR CHART
        # --------------------------------------------------

        with left:

            st.plotly_chart(
                charts.class_bar_chart(class_counts),
                use_container_width=True,
                key="webcam_bar_chart"
            )

        # --------------------------------------------------
        # PIE CHART
        # --------------------------------------------------

        with right:

            st.plotly_chart(
                charts.class_pie_chart(class_counts),
                use_container_width=True,
                key="webcam_pie_chart"
            )

        # --------------------------------------------------
        # FPS CHART
        # --------------------------------------------------

        st.plotly_chart(
            charts.fps_chart(fps_history),
            use_container_width=True,
            key="webcam_fps_chart"
        )

    else:

        st.info("Waiting for detections...")

# ==========================================================
# SNAPSHOT
# ==========================================================

st.divider()

st.subheader("📸 Snapshot")

col1, col2 = st.columns(2)

with col1:

    if st.button("📷 Save Current Frame"):

        with shared.lock:

            frame = shared.last_frame

        if frame is not None:

            import os
            import cv2
            from datetime import datetime
            from config import OUTPUT_DIR

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            filename = os.path.join(
                OUTPUT_DIR,
                f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            cv2.imwrite(filename, frame)

            st.success("Snapshot saved successfully!")

            with open(filename, "rb") as file:

                st.download_button(
                    "⬇ Download Snapshot",
                    data=file,
                    file_name=os.path.basename(filename),
                    mime="image/jpeg",
                )

        else:

            st.warning("No frame available.")

with col2:

    if st.button("🗑 Reset Statistics"):

        with shared.lock:

            shared.fps = 0
            shared.objects = 0
            shared.confidence = 0
            shared.inference = 0
            shared.class_counts = {}
            shared.detections = []

        st.session_state.fps_history = []

        st.success("Statistics reset successfully.")

# ==========================================================
# DETECTION HISTORY
# ==========================================================

st.divider()

st.subheader("📚 Detection History")

if len(detections):

    history_df = pd.DataFrame(detections)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info("No detections available.")
# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

st.divider()

st.subheader("💻 System Information")

left, right = st.columns(2)

with left:

    st.info("🤖 Model : YOLOv8")

    st.info("🎯 Tracker : ByteTrack")

    st.info("🐍 Framework : Streamlit + WebRTC")

with right:

    status = "🟢 Running" if ctx.state.playing else "🔴 Stopped"

    st.info(f"Camera : {status}")

    st.info(f"Detected Objects : {objects}")

    st.info(f"FPS : {fps:.2f}")

# ==========================================================
# LIVE STATUS
# ==========================================================

st.divider()

st.subheader("📡 Live Status")

c1, c2, c3 = st.columns(3)

with c1:

    if ctx.state.playing:
        st.success("🟢 Webcam Connected")
    else:
        st.error("🔴 Webcam Stopped")

with c2:

    if objects > 0:
        st.success(f"🎯 {objects} object(s)")
    else:
        st.warning("No object detected")

with c3:

    st.info(f"⚡ FPS : {fps:.2f}")

# ==========================================================
# ABOUT
# ==========================================================

st.divider()

with st.expander("ℹ About this Project"):

    st.markdown("""
## AI Object Detection & Tracking

### Features

- 📷 Live Webcam Detection
- 🤖 YOLOv8 Object Detection
- 🎯 ByteTrack Multi-Object Tracking
- 📊 Live Dashboard
- 📈 Plotly Charts
- 📋 Detection Table
- 📸 Snapshot Capture
- ⚡ FPS Monitoring
- ⏱ Inference Time
- 📚 Detection History

### Technologies

- Python
- Streamlit
- streamlit-webrtc
- OpenCV
- Ultralytics YOLOv8
- Plotly
- Pandas
""")

# ==========================================================
# AUTO REFRESH
# ==========================================================

from streamlit_autorefresh import st_autorefresh

st_autorefresh(
    interval=1000,
    key="webcam_refresh"
)

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
"""
---
<div style="text-align:center">

## 🎯 AI Object Detection & Tracking

Developed with ❤️ using

**YOLOv8 • ByteTrack • Streamlit • WebRTC • OpenCV • Plotly**

© 2026 Atimad BEL CAID

</div>
""",
unsafe_allow_html=True,
)