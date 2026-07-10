"""
=========================================================
Settings
---------------------------------------------------------
AI Object Detection & Tracking

Author : Atimad BEL CAID
=========================================================
"""

import streamlit as st

from config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER,
    VERSION,
    AUTHOR
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Application Settings")

st.markdown(
"""
Configure the Object Detection application.
"""
)

st.divider()

# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {

    "confidence_threshold": CONFIDENCE_THRESHOLD,

    "iou_threshold": IOU_THRESHOLD,

    "image_size": IMAGE_SIZE,

    "tracker": TRACKER,

    "save_results": True,

    "show_boxes": True,

    "show_labels": True,

    "show_confidence": True,

    "theme": "Dark"

}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value

# ==========================================================
# MODEL SETTINGS
# ==========================================================

st.header("🤖 Model")

col1, col2 = st.columns(2)

with col1:

    st.text_input(
        "Model Path",
        MODEL_PATH,
        disabled=True
    )

    st.slider(
        "Confidence Threshold",
        0.10,
        1.00,
        st.session_state.confidence_threshold,
        0.05,
        key="confidence_threshold"
    )

    st.slider(
        "IOU Threshold",
        0.10,
        1.00,
        st.session_state.iou_threshold,
        0.05,
        key="iou_threshold"
    )

with col2:

    st.selectbox(
        "Image Size",
        [320, 416, 512, 640, 960, 1280],
        index=[320,416,512,640,960,1280].index(st.session_state.image_size),
        key="image_size"
    )

    st.selectbox(
        "Tracker",
        [
            "bytetrack.yaml",
            "botsort.yaml"
        ],
        index=0 if st.session_state.tracker=="bytetrack.yaml" else 1,
        key="tracker"
    )

st.divider()

# ==========================================================
# DISPLAY SETTINGS
# ==========================================================

st.header("🎨 Display")

left, right = st.columns(2)

with left:

    st.checkbox(
        "Show Bounding Boxes",
        key="show_boxes"
    )

    st.checkbox(
        "Show Labels",
        key="show_labels"
    )

with right:

    st.checkbox(
        "Show Confidence",
        key="show_confidence"
    )

    st.checkbox(
        "Save Results Automatically",
        key="save_results"
    )

st.divider()

# ==========================================================
# THEME
# ==========================================================

st.header("🌙 Theme")

st.radio(

    "Application Theme",

    ["Dark", "Light"],

    key="theme",

    horizontal=True

)

st.divider()

# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

st.header("💻 System Information")

info = {

    "Version": VERSION,

    "Author": AUTHOR,

    "Framework": "Streamlit",

    "Detector": "YOLOv8",

    "Tracker": st.session_state.tracker,

    "Image Size": st.session_state.image_size,

    "Model": MODEL_PATH

}

for key, value in info.items():

    st.info(f"**{key} :** {value}")

st.divider()

# ==========================================================
# SAVE SETTINGS
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:

    if st.button(
        "💾 Save Settings",
        use_container_width=True
    ):

        st.success(
            "Settings saved successfully."
        )

with col2:

    if st.button(
        "🔄 Reset Defaults",
        use_container_width=True
    ):

        st.session_state.confidence_threshold = CONFIDENCE_THRESHOLD
        st.session_state.iou_threshold = IOU_THRESHOLD
        st.session_state.image_size = IMAGE_SIZE
        st.session_state.tracker = TRACKER
        st.session_state.show_boxes = True
        st.session_state.show_labels = True
        st.session_state.show_confidence = True
        st.session_state.save_results = True
        st.session_state.theme = "Dark"

        st.success(
            "Default settings restored."
        )

with col3:

    if st.button(
        "🗑 Clear Session",
        use_container_width=True
    ):

        keys = list(st.session_state.keys())

        for key in keys:

            del st.session_state[key]

        st.success(
            "Session cleared successfully."
        )

st.divider()

# ==========================================================
# CURRENT SETTINGS
# ==========================================================

st.header("📋 Current Configuration")

config = {

    "Confidence Threshold":
        st.session_state.confidence_threshold,

    "IOU Threshold":
        st.session_state.iou_threshold,

    "Image Size":
        st.session_state.image_size,

    "Tracker":
        st.session_state.tracker,

    "Theme":
        st.session_state.theme,

    "Bounding Boxes":
        st.session_state.show_boxes,

    "Labels":
        st.session_state.show_labels,

    "Confidence":
        st.session_state.show_confidence,

    "Save Results":
        st.session_state.save_results

}

st.json(config)

st.divider()

# ==========================================================
# ABOUT
# ==========================================================

with st.expander("ℹ About"):

    st.markdown("""
### AI Object Detection & Tracking

**Features**

- YOLOv8 Object Detection
- ByteTrack Tracking
- Image Detection
- Video Detection
- Webcam Detection
- Dashboard
- Plotly Charts
- CSV Export
- Performance Monitoring

---

Developed using:

- Python
- Streamlit
- OpenCV
- Ultralytics
- Plotly
- Pandas
- NumPy
""")

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
---
<div style="text-align:center">

### ⚙️ Settings

AI Object Detection & Tracking

**YOLOv8 • ByteTrack • Streamlit**

© 2026 Atimad BEL CAID

</div>
""",
unsafe_allow_html=True
)