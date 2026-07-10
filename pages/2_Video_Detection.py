"""
=========================================================
Video Detection
---------------------------------------------------------
YOLOv8 Object Detection & Tracking

Author : Atimad BEL CAID
=========================================================
"""

import os
import cv2
import time
import tempfile
import streamlit as st
import pandas as pd

from detector import ObjectDetector
from utils.metrics import Metrics
from utils.charts import DashboardCharts
from utils.helpers import bgr_to_rgb
from config import OUTPUT_DIR

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Video Detection",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Video Object Detection")

st.markdown("""
Upload a video and perform **YOLOv8 + ByteTrack** object detection and tracking.
""")

st.divider()

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_detector():
    return ObjectDetector()

detector = load_detector()
metrics = Metrics()
charts = DashboardCharts()

# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "fps": 0,
    "objects": 0,
    "confidence": 0,
    "inference": 0,
    "class_counts": {},
    "detections": [],
    "fps_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.40,
    0.05
)

show_table = st.sidebar.checkbox(
    "Show Detection Table",
    True
)

show_charts = st.sidebar.checkbox(
    "Show Charts",
    True
)

show_fps = st.sidebar.checkbox(
    "Show FPS Graph",
    True
)

# ==========================================================
# VIDEO UPLOAD
# ==========================================================

uploaded_video = st.file_uploader(
    "📂 Upload a Video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ==========================================================
# PLACEHOLDERS
# ==========================================================

progress_bar = st.empty()

video_placeholder = st.empty()

metrics_placeholder = st.empty()

table_placeholder = st.empty()

charts_placeholder = st.empty()

download_placeholder = st.empty()

# ==========================================================
# PROCESS VIDEO
# ==========================================================

if uploaded_video is not None:

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(uploaded_video.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    if fps_video <= 0:
        fps_video = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "processed_video.mp4"
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_video,
        (width, height)
    )

    fps_history = []

    frame_number = 0
    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        start = time.time()

        # ------------------------------------------------------
        # YOLO DETECTION
        # ------------------------------------------------------

        annotated, detections = detector.track(
            frame,
            confidence
        )

        writer.write(annotated)

        # ------------------------------------------------------
        # METRICS
        # ------------------------------------------------------

        inference = metrics.update_inference(start)

        fps = metrics.update_fps()

        total = metrics.total_objects(detections)

        avg = metrics.average_confidence(detections)

        class_counts = metrics.class_counter(detections)

        dataframe = metrics.dataframe(detections)

        # ------------------------------------------------------
        # SESSION STATE
        # ------------------------------------------------------

        st.session_state.fps = fps
        st.session_state.objects = total
        st.session_state.confidence = avg
        st.session_state.inference = inference
        st.session_state.class_counts = class_counts
        st.session_state.detections = detections

        st.session_state.fps_history.append(fps)

        if len(st.session_state.fps_history) > 100:
            st.session_state.fps_history.pop(0)

        fps_history = st.session_state.fps_history

        # ------------------------------------------------------
        # PROGRESS BAR
        # ------------------------------------------------------

        progress_bar.progress(frame_number / total_frames)

        # ------------------------------------------------------
        # VIDEO DISPLAY
        # ------------------------------------------------------

        video_placeholder.image(
            bgr_to_rgb(annotated),
            channels="RGB",
            use_container_width=True
        )

        # ------------------------------------------------------
        # LIVE METRICS
        # ------------------------------------------------------

        with metrics_placeholder.container():

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "🎯 Objects",
                total
            )

            c2.metric(
                "⚡ FPS",
                f"{fps:.2f}"
            )

            c3.metric(
                "🧠 Confidence",
                f"{avg:.2f}%"
            )

            c4.metric(
                "⏱ Inference",
                f"{inference:.2f} ms"
            )

        # ------------------------------------------------------
        # DETECTION TABLE
        # ------------------------------------------------------

        if show_table:

            with table_placeholder.container():

                st.subheader("📋 Current Detections")

                if not dataframe.empty:

                    st.dataframe(
                        dataframe,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info("No object detected.")
        # ------------------------------------------------------
        # CHARTS
        # ------------------------------------------------------

        if show_charts:

            with charts_placeholder.container():

                if len(class_counts):

                    left, right = st.columns(2)

                    with left:

                        st.plotly_chart(
                            charts.class_bar_chart(class_counts),
                            use_container_width=True,
                            key=f"video_bar_{frame_number}"
                        )

                    with right:

                        st.plotly_chart(
                            charts.class_pie_chart(class_counts),
                            use_container_width=True,
                            key=f"video_pie_{frame_number}"
                        )

                if show_fps and len(fps_history) > 1:

                    st.plotly_chart(
                        charts.fps_chart(fps_history),
                        use_container_width=True,
                        key=f"video_fps_{frame_number}"
                    )

        # ------------------------------------------------------
        # SMALL DELAY
        # ------------------------------------------------------

        time.sleep(0.001)

    # ==========================================================
    # END OF VIDEO
    # ==========================================================

    cap.release()
    writer.release()

    progress_bar.empty()

    st.success("✅ Video processing completed successfully!")

    # ==========================================================
    # VIDEO PREVIEW
    # ==========================================================

    st.subheader("🎬 Processed Video")

    st.video(output_path)

    # ==========================================================
    # DOWNLOAD
    # ==========================================================

    with open(output_path, "rb") as video_file:

        download_placeholder.download_button(
            label="⬇ Download Processed Video",
            data=video_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # ==========================================================
    # FINAL STATISTICS
    # ==========================================================

    st.divider()

    st.subheader("📊 Final Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Objects",
        st.session_state.objects
    )

    col2.metric(
        "FPS",
        f"{st.session_state.fps:.2f}"
    )

    col3.metric(
        "Confidence",
        f"{st.session_state.confidence:.2f}%"
    )

    col4.metric(
        "Inference",
        f"{st.session_state.inference:.2f} ms"
    )
# ==========================================================
# CLEANUP
# ==========================================================

try:
    os.remove(temp_video.name)
except Exception:
    pass

# ==========================================================
# HISTORY
# ==========================================================

st.divider()

st.subheader("📚 Detection History")

history = st.session_state.get("detections", [])

if history:

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No detection history available.")

# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

st.divider()

st.subheader("💻 System Information")

left, right = st.columns(2)

with left:

    st.info("🤖 Model : YOLOv8n")

    st.info("🎯 Tracker : ByteTrack")

    st.info("🖥 Framework : Streamlit")

with right:

    st.info("🎥 Input : Video File")

    st.info("📦 OpenCV : Enabled")

    st.info("📊 Plotly Dashboard : Enabled")

# ==========================================================
# PERFORMANCE SUMMARY
# ==========================================================

st.divider()

st.subheader("📈 Performance Summary")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Objects",
        st.session_state.objects
    )

with c2:

    st.metric(
        "FPS",
        f"{st.session_state.fps:.2f}"
    )

with c3:

    st.metric(
        "Inference",
        f"{st.session_state.inference:.2f} ms"
    )

# ==========================================================
# ABOUT
# ==========================================================

st.divider()

with st.expander("ℹ About this Application"):

    st.markdown("""
### AI Object Detection & Tracking

This application provides:

- 🎯 YOLOv8 Object Detection
- 🚀 ByteTrack Multi Object Tracking
- 🎥 Video Processing
- 📈 Live Statistics
- 📊 Interactive Plotly Charts
- 📋 Detection Table
- 💾 Processed Video Download
- ⚡ Performance Monitoring

---

### Technologies

- Python
- Streamlit
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- Plotly
- Pandas
""")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
"""
<div style='text-align:center'>

## 🎯 AI Object Detection & Tracking

Developed with ❤️ using

### YOLOv8 • ByteTrack • Streamlit • OpenCV • Plotly

© 2026 Atimad BEL CAID

</div>
""",
unsafe_allow_html=True
)