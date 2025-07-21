# 🪑 Chair Occupancy Tracker

Real-time chair utilization analysis using computer vision.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## Features
- YouTube video processing
- Chair detection and tracking
- Occupancy analysis using pose estimation
- Utilization metrics and visualization
- Annotated video output

## How to Use
1. Enter YouTube URL or upload MP4
2. Click "Process Video"
3. View results in dashboard
4. Generate tracking video

## Local Installation
```bash
git clone https://github.com/your-username/chair-occupancy-tracker.git
cd chair-occupancy-tracker
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run dashboard.py
```

## Deployment
Deployed on Streamlit Community Cloud:
`https://your-app-name.streamlit.app`
