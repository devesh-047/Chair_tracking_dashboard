import os
import cv2
import subprocess

def download_youtube_video(url, cookies, output_path):
    """Download YouTube video using yt-dlp with cookies authentication"""
    try:
        subprocess.run([
            'yt-dlp', 
            '--cookies', cookies,
            '-f', 'best[ext=mp4]',
            '-o', output_path,
            url
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        return False

def extract_frames(video_path, output_folder, frame_prefix="frame"):
    """Extract frames from video using OpenCV"""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_filename = os.path.join(
            output_folder, 
            f"{frame_prefix}_{frame_count:05d}.jpg"
        )
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    return frame_count

def create_tracking_video(input_path, chair_tracks, output_path):
    """Create video with chair tracking annotations"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get chairs for this frame
        chairs = chair_tracks.get(frame_count, [])
        
        # Draw annotations
        for chair in chairs:
            chair_id = chair['id']
            x1, y1, x2, y2 = chair['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw chair ID
            cv2.putText(frame, f"Chair-{chair_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Write frame to output
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    return True