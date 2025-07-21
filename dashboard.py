import streamlit as st
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from core.chair_tracker import ChairTracker
from core.video_processor import download_youtube_video, extract_frames, create_tracking_video
from core.pose_analyzer import PoseAnalyzer
from utils.visualization import visualize_frame_with_chairs, plot_detected_vs_occupied
import tempfile

# Page setup
st.set_page_config(layout="wide", page_title="Chair Occupancy Tracker")
st.title("🪑 Chair Occupancy Tracker")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # Video source selection
    video_source = st.radio("Video Source", ["YouTube URL", "Upload MP4"])
    
    if video_source == "YouTube URL":
        video_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=PBKvLfUOKaI")
        cookies_path = st.text_input("Cookies Path", "cookies.txt")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])
        video_url = None
        cookies_path = None
    
    output_dir = st.text_input("Output Directory", "data")
    process_btn = st.button("Process Video")
    
    st.header("Visualization")
    frame_num = st.slider("Select Frame", 0, 300, 125)
    
    # Video generation
    if os.path.exists(output_dir):
        st.header("Video Output")
        generate_video_btn = st.button("Generate Tracking Video")
    
    # Display tracked chair IDs
    if os.path.exists(os.path.join(output_dir, "chair_tracks.pkl")):
        with open(os.path.join(output_dir, "chair_tracks.pkl"), "rb") as f:
            chair_tracks = pickle.load(f)
        all_chair_ids = sorted({chair['id'] for frame_data in chair_tracks.values() for chair in frame_data})
        st.header("Tracked Chair IDs")
        st.write(f"Total chairs tracked: {len(all_chair_ids)}")
        st.write(all_chair_ids)

# Processing pipeline
if process_btn:
    with st.status("Processing video...", expanded=True) as status:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if video_source == "YouTube URL" and video_url:
            # Download YouTube video
            video_path = os.path.join(output_dir, "downloaded_video.mp4")
            st.write("Downloading video...")
            download_youtube_video(video_url, cookies_path, video_path)
        elif uploaded_file:
            # Save uploaded video
            video_path = os.path.join(output_dir, "uploaded_video.mp4")
            st.write("Saving uploaded video...")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            st.error("Please provide a video source")
            st.stop()
        
        # Extract frames
        frames_dir = os.path.join(output_dir, "video_frames")
        st.write("Extracting frames...")
        frame_count = extract_frames(video_path, frames_dir)
        
        # Track chairs
        st.write("Tracking chairs...")
        tracker = ChairTracker()
        chair_tracks = tracker.track_chairs(
            video_path,
            output_path=os.path.join(output_dir, "chair_tracks.pkl")
        )
        
        # Analyze poses
        st.write("Analyzing poses...")
        analyzer = PoseAnalyzer()
        results = {}
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        
        for i, fname in enumerate(frame_files[::5]):
            frame_number = i * 5
            frame_path = os.path.join(frames_dir, fname)
            chairs, seated, occupied_ids = analyzer.analyze_frame(
                frame_path, frame_number, chair_tracks
            )
            results[frame_number] = {
                "chairs": chairs,
                "seated": seated,
                "occupied_ids": occupied_ids
            }
        
        # Save results
        with open(os.path.join(output_dir, "analysis_results.pkl"), "wb") as f:
            pickle.dump(results, f)
        
        status.update(label="Processing complete!", state="complete")

# Visualization section
if os.path.exists(output_dir):
    # Load data
    chair_tracks_path = os.path.join(output_dir, "chair_tracks.pkl")
    analysis_path = os.path.join(output_dir, "analysis_results.pkl")
    frames_dir = os.path.join(output_dir, "video_frames")
    
    if os.path.exists(chair_tracks_path) and os.path.exists(analysis_path):
        with open(chair_tracks_path, "rb") as f:
            chair_tracks = pickle.load(f)
        
        with open(analysis_path, "rb") as f:
            analysis_results = pickle.load(f)
        
        # Show frame visualization
        frame_path = os.path.join(frames_dir, f"frame_{frame_num:05d}.jpg")
        if os.path.exists(frame_path):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Frame {frame_num}")
                # Mark occupied chairs
                occupied_ids = analysis_results.get(frame_num, {}).get("occupied_ids", [])
                for chair in chair_tracks.get(frame_num, []):
                    chair["occupied"] = chair["id"] in occupied_ids
                
                fig = visualize_frame_with_chairs(frame_path, chair_tracks.get(frame_num, []))
                st.pyplot(fig)
                
            with col2:
                st.subheader("Occupancy Analysis")
                
                # Show metrics
                current_chairs = analysis_results.get(frame_num, {}).get("chairs", 0)
                current_occupied = analysis_results.get(frame_num, {}).get("seated", 0)
                occupied_ids = analysis_results.get(frame_num, {}).get("occupied_ids", [])
                
                st.metric("Total Chairs", current_chairs)
                st.metric("Occupied Chairs", current_occupied)
                st.metric("Utilization", f"{current_occupied/current_chairs*100:.1f}%")
                
                # Show occupied chair IDs
                st.subheader("Occupied Chair IDs")
                if occupied_ids:
                    st.write(occupied_ids)
                else:
                    st.write("No chairs occupied")
                
                # Show all tracked chair IDs
                st.subheader("All Tracked Chair IDs")
                all_chair_ids = sorted({chair['id'] for frame_data in chair_tracks.values() for chair in frame_data})
                st.write(f"Total: {len(all_chair_ids)} chairs")
                st.write(all_chair_ids)
        else:
            st.warning(f"Frame {frame_num} not found in {frames_dir}")
        
        # NEW: Detected vs Occupied chairs graph
        st.subheader("Detected Chairs vs Occupied Chairs")
        fig = plot_detected_vs_occupied(analysis_results)
        st.pyplot(fig)
    else:
        st.warning("Processed data not found. Please process a video first.")
else:
    st.info("Configure settings and click 'Process Video' to begin")

# Video generation
if 'generate_video_btn' in locals() and generate_video_btn:
    video_path = os.path.join(output_dir, "downloaded_video.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(output_dir, "uploaded_video.mp4")
    
    chair_tracks_path = os.path.join(output_dir, "chair_tracks.pkl")
    
    if os.path.exists(video_path) and os.path.exists(chair_tracks_path):
        with st.status("Generating tracking video...", expanded=True) as status:
            st.write("Creating video with tracking annotations...")
            with open(chair_tracks_path, "rb") as f:
                chair_tracks = pickle.load(f)
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
                output_video_path = tmpfile.name
            
            # Generate video
            create_tracking_video(video_path, chair_tracks, output_video_path)
            
            st.write("Video generation complete!")
            status.update(label="Video ready!", state="complete")
            
            # Show video
            st.subheader("Tracking Video")
            st.video(output_video_path)
            
            # Download button
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="Download Tracking Video",
                data=video_bytes,
                file_name="chair_tracking_video.mp4",
                mime="video/mp4"
            )
    else:
        st.error("Required files not found. Please process a video first.")