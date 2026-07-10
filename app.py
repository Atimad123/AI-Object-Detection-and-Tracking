"""
=========================================================
AI Object Detection & Tracking
---------------------------------------------------------
Main Application

Author : Atimad BEL CAID
=========================================================
"""

import streamlit as st
from pathlib import Path

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD CSS
# ==========================================================

css_file = Path("assets/style.css")

if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================================
# INITIALIZE SESSION STATE
# ==========================================================

defaults = {
    "fps": 0.0,
    "objects": 0,
    "confidence": 0.0,
    "inference": 0.0,
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

st.sidebar.image(
    "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg",
    width=220
)

st.sidebar.title("🎯 AI Object Detection")

st.sidebar.success("Navigation")

st.sidebar.markdown("""
Utilise le menu **Pages** de Streamlit pour accéder à :

- 🖼 Image Detection
- 🎥 Video Detection
- 📷 Webcam
- 📊 Dashboard
- ⚙ Settings
""")

st.sidebar.divider()

st.sidebar.markdown("### 📈 Live Statistics")

st.sidebar.metric(
    "Objects",
    st.session_state.objects
)

st.sidebar.metric(
    "FPS",
    f"{st.session_state.fps:.2f}"
)

st.sidebar.metric(
    "Confidence",
    f"{st.session_state.confidence:.2f}%"
)

st.sidebar.metric(
    "Inference",
    f"{st.session_state.inference:.2f} ms"
)

st.sidebar.divider()

st.sidebar.info(
    "YOLOv8 + ByteTrack\n\n"
    "Powered by Streamlit"
)

# ==========================================================
# MAIN PAGE
# ==========================================================

st.title("🎯 AI Object Detection & Tracking")

st.markdown(
"""
Bienvenue dans une application professionnelle de détection
et suivi d'objets basée sur **YOLOv8** et **ByteTrack**.

Sélectionne une page depuis le menu de gauche pour commencer.
"""
)

st.divider()

# ==========================================================
# FEATURES
# ==========================================================

st.subheader("🚀 Fonctionnalités")

col1, col2 = st.columns(2)

with col1:

    st.success("🖼 Détection sur image")

    st.success("🎥 Détection sur vidéo")

    st.success("📷 Détection Webcam")

    st.success("🎯 Multi Object Tracking")

with col2:

    st.success("📊 Dashboard interactif")

    st.success("📈 Graphiques Plotly")

    st.success("📥 Export CSV")

    st.success("⚙ Paramètres personnalisables")

st.divider()

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

st.subheader("🧠 Technologies")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.info("Python")

with tech2:
    st.info("YOLOv8")

with tech3:
    st.info("OpenCV")

with tech4:
    st.info("Streamlit")

st.divider()

# ==========================================================
# APPLICATION STATUS
# ==========================================================

st.subheader("📋 État de l'application")

status1, status2, status3, status4 = st.columns(4)

with status1:
    st.metric("Objects", st.session_state.objects)

with status2:
    st.metric("FPS", f"{st.session_state.fps:.2f}")

with status3:
    st.metric("Confidence", f"{st.session_state.confidence:.2f}%")

with status4:
    st.metric("Inference", f"{st.session_state.inference:.2f} ms")

st.divider()

# ==========================================================
# PROJECT STRUCTURE
# ==========================================================

with st.expander("📁 Structure du projet"):

    st.code(
"""
Object_Detection_Tracking/
│
├── app.py
├── config.py
├── detector.py
├── requirements.txt
│
├── assets/
│   ├── style.css
│   ├── images/
│   ├── videos/
│   └── output/
│
├── models/
│   └── yolov8n.pt
│
├── pages/
│   ├── 1_Image_Detection.py
│   ├── 2_Video_Detection.py
│   ├── 3_Webcam.py
│   ├── 4_Dashboard.py
│   └── 5_Settings.py
│
├── utils/
│   ├── helpers.py
│   ├── metrics.py
│   └── charts.py
"""
    )

st.divider()

# ==========================================================
# ABOUT
# ==========================================================

with st.expander("ℹ À propos"):

    st.markdown("""
### AI Object Detection & Tracking

Cette application permet :

- Détection d'objets avec YOLOv8
- Tracking multi-objets avec ByteTrack
- Détection sur images
- Détection sur vidéos
- Webcam en temps réel
- Dashboard interactif
- Graphiques Plotly
- Export des résultats
- Paramètres personnalisables
""")

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
---
<div style="text-align:center">

# 🎯 AI Object Detection & Tracking

Développé avec ❤️ par **Atimad BEL CAID**

### YOLOv8 • ByteTrack • Streamlit • OpenCV • Plotly

</div>
""",
unsafe_allow_html=True
)