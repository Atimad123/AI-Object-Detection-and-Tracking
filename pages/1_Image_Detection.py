"""
=========================================================
Image Detection
---------------------------------------------------------
YOLOv8 Image Detection

Author : Atimad BEL CAID
=========================================================
"""

import os
import cv2
import tempfile
import streamlit as st

from detector import ObjectDetector
from utils.metrics import Metrics
from utils.charts import DashboardCharts
from utils.helpers import bgr_to_rgb
from config import OUTPUT_DIR

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Image Detection",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Object Detection")

st.markdown(
"""
Upload an image and detect objects using **YOLOv8**.
"""
)

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

if "fps" not in st.session_state:
    st.session_state.fps = 0

if "objects" not in st.session_state:
    st.session_state.objects = 0

if "confidence" not in st.session_state:
    st.session_state.confidence = 0

if "inference" not in st.session_state:
    st.session_state.inference = 0

if "class_counts" not in st.session_state:
    st.session_state.class_counts = {}

if "detections" not in st.session_state:
    st.session_state.detections = []

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence",
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

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

# ==========================================================
# PROCESS IMAGE
# ==========================================================

if uploaded_image is not None:

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    temp_file.write(uploaded_image.read())

    temp_file.close()

    image = cv2.imread(temp_file.name)

    with st.spinner("Running YOLOv8 Detection..."):

        annotated, detections = detector.detect(
            image,
            confidence
        )

    total = metrics.total_objects(detections)

    avg = metrics.average_confidence(detections)

    class_counts = metrics.class_counter(detections)

    dataframe = metrics.dataframe(detections)

    # ======================================================
    # UPDATE DASHBOARD
    # ======================================================

    st.session_state.objects = total
    st.session_state.confidence = avg
    st.session_state.class_counts = class_counts
    st.session_state.detections = detections

    # ======================================================
    # SHOW IMAGES
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            bgr_to_rgb(image),
            use_container_width=True
        )

    with col2:

        st.subheader("Detection Result")

        st.image(
            bgr_to_rgb(annotated),
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # METRICS
    # ======================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Detected Objects",
        total
    )

    c2.metric(
        "Average Confidence",
        f"{avg:.2f}%"
    )

    c3.metric(
        "Classes",
        len(class_counts)
    )

    st.divider()

    # ======================================================
    # TABLE
    # ======================================================

    if show_table:

        st.subheader("Detection Table")

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

    # ======================================================
    # CHARTS
    # ======================================================

    if show_charts and len(class_counts):

        left, right = st.columns(2)

        with left:

            st.plotly_chart(
                charts.class_bar_chart(class_counts),
                use_container_width=True
            )

        with right:

            st.plotly_chart(
                charts.class_pie_chart(class_counts),
                use_container_width=True
            )

        st.divider()

    # ======================================================
    # SAVE IMAGE
    # ======================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        "detected_image.jpg"
    )

    cv2.imwrite(
        output_path,
        annotated
    )

    with open(output_path, "rb") as file:

        st.download_button(
            label="⬇ Download Result",
            data=file,
            file_name="detected_image.jpg",
            mime="image/jpeg"
        )

    os.remove(temp_file.name)

else:

    st.info("Please upload an image to start detection.")