"""
=========================================================
Dashboard
---------------------------------------------------------
AI Object Detection & Tracking Dashboard

Author : Atimad BEL CAID
=========================================================
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.charts import DashboardCharts

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Object Detection Dashboard")
st.caption("Real-Time Analytics for YOLOv8 Object Detection")

st.divider()

# ==========================================================
# INITIALIZE SESSION STATE
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
# DASHBOARD CHARTS
# ==========================================================

charts = DashboardCharts()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙ Dashboard Settings")

auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=True
)

show_table = st.sidebar.checkbox(
    "Show Detection Table",
    value=True
)

show_charts = st.sidebar.checkbox(
    "Show Charts",
    value=True
)

# ==========================================================
# GET DATA
# ==========================================================

fps = st.session_state.fps
objects = st.session_state.objects
confidence = st.session_state.confidence
inference = st.session_state.inference
class_counts = st.session_state.class_counts
detections = st.session_state.detections

# ==========================================================
# FPS HISTORY
# ==========================================================

st.session_state.fps_history.append(fps)

if len(st.session_state.fps_history) > 100:
    st.session_state.fps_history.pop(0)

fps_history = st.session_state.fps_history

# ==========================================================
# KPI CARDS
# ==========================================================

st.subheader("📈 Live Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="🎯 Objects",
        value=objects
    )

with col2:

    st.metric(
        label="⚡ FPS",
        value=f"{fps:.2f}"
    )

with col3:

    st.metric(
        label="🧠 Confidence",
        value=f"{confidence:.2f}%"
    )

with col4:

    st.metric(
        label="⏱ Inference",
        value=f"{inference:.2f} ms"
    )

st.divider()
# ==========================================================
# CHARTS
# ==========================================================

if show_charts:

    st.subheader("📊 Detection Analytics")

    if class_counts:

        left, right = st.columns(2)

        # --------------------------------------------------
        # BAR CHART
        # --------------------------------------------------

        with left:

            st.plotly_chart(
                charts.class_bar_chart(class_counts),
                use_container_width=True
            )

        # --------------------------------------------------
        # PIE CHART
        # --------------------------------------------------

        with right:

            st.plotly_chart(
                charts.class_pie_chart(class_counts),
                use_container_width=True
            )

    else:

        st.info("No object statistics available yet.")

# ==========================================================
# FPS HISTORY
# ==========================================================

st.subheader("📈 FPS History")

if len(fps_history) > 1:

    st.plotly_chart(
        charts.fps_chart(fps_history),
        use_container_width=True
    )

else:

    st.info("Waiting for FPS data...")

# ==========================================================
# SUMMARY
# ==========================================================

st.subheader("📋 Detection Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    st.success(f"🎯 Total Objects Detected : {objects}")

    st.success(f"🧠 Average Confidence : {confidence:.2f}%")

with summary_col2:

    st.success(f"⚡ Current FPS : {fps:.2f}")

    st.success(f"⏱ Average Inference : {inference:.2f} ms")

st.divider()
# ==========================================================
# DETECTION TABLE
# ==========================================================

if show_table:

    st.subheader("📋 Detection Table")

    if detections:

        df = pd.DataFrame(detections)

        # ---------------------------------------------
        # FILTER BY CLASS
        # ---------------------------------------------

        if "class" in df.columns:

            classes = ["All"] + sorted(df["class"].unique().tolist())

            selected = st.selectbox(
                "Filter by Class",
                classes
            )

            if selected != "All":
                df = df[df["class"] == selected]

        # ---------------------------------------------
        # SEARCH
        # ---------------------------------------------

        search = st.text_input(
            "🔍 Search Object"
        )

        if search:

            search = search.lower()

            df = df[
                df.astype(str)
                  .apply(lambda x: x.str.lower())
                  .apply(lambda x: x.str.contains(search))
                  .any(axis=1)
            ]

        # ---------------------------------------------
        # TABLE
        # ---------------------------------------------

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------------------
        # DOWNLOAD CSV
        # ---------------------------------------------

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name="detections.csv",
            mime="text/csv"
        )

    else:

        st.info("No detections available.")

st.divider()

# ==========================================================
# CLASS SUMMARY
# ==========================================================

st.subheader("📊 Object Summary")

if class_counts:

    summary_df = pd.DataFrame({
        "Class": list(class_counts.keys()),
        "Count": list(class_counts.values())
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning("No class statistics available.")

st.divider()

# ==========================================================
# RESET BUTTON
# ==========================================================

left, right = st.columns([1, 5])

with left:

    if st.button("🗑 Reset"):

        st.session_state.fps = 0
        st.session_state.objects = 0
        st.session_state.confidence = 0
        st.session_state.inference = 0
        st.session_state.class_counts = {}
        st.session_state.detections = []
        st.session_state.fps_history = []

        st.success("Dashboard reset successfully.")

with right:

    st.info("Reset clears all current dashboard statistics.")
# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

st.divider()

st.subheader("💻 System Information")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🤖 AI Model")

    st.info("""
**Model :** YOLOv8n

**Tracker :** ByteTrack

**Framework :** Ultralytics

**Language :** Python 3
""")

with col2:

    st.markdown("### 🖥 Environment")

    st.info(f"""
**Dashboard Updated :**
{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

**Application :**
Object Detection & Tracking

**Interface :**
Streamlit
""")

st.divider()

# ==========================================================
# APPLICATION FEATURES
# ==========================================================

st.subheader("🚀 Application Features")

feature1, feature2 = st.columns(2)

with feature1:

    st.success("✅ Image Detection")

    st.success("✅ Video Detection")

    st.success("✅ Webcam Detection")

    st.success("✅ Multi-Object Tracking")

with feature2:

    st.success("✅ Live Dashboard")

    st.success("✅ Plotly Charts")

    st.success("✅ CSV Export")

    st.success("✅ Dark Theme")

st.divider()

# ==========================================================
# PERFORMANCE
# ==========================================================

st.subheader("📈 Performance")

perf1, perf2, perf3 = st.columns(3)

with perf1:
    st.metric(
        "Objects",
        objects
    )

with perf2:
    st.metric(
        "FPS",
        f"{fps:.2f}"
    )

with perf3:
    st.metric(
        "Inference",
        f"{inference:.2f} ms"
    )

st.divider()

# ==========================================================
# ABOUT
# ==========================================================

with st.expander("ℹ About this Project"):

    st.markdown("""
# AI Object Detection & Tracking

This application provides:

- YOLOv8 Object Detection
- ByteTrack Multi Object Tracking
- Image Detection
- Video Detection
- Webcam Detection
- Real-Time Dashboard
- Interactive Plotly Charts
- Detection History
- CSV Export
- Performance Monitoring

---

### Technologies

- Python
- Streamlit
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- Plotly
- Pandas
- NumPy
""")

st.divider()

# ==========================================================
# AUTO REFRESH
# ==========================================================

if auto_refresh:

    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(
        interval=2000,
        key="dashboard_refresh"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
---
<div style="text-align:center">

## 🎯 AI Object Detection & Tracking

Developed with ❤️ using

### YOLOv8 • ByteTrack • Streamlit • OpenCV • Plotly

© 2026 Atimad BEL CAID

</div>
""",
unsafe_allow_html=True
)