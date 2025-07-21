import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from .config import config

# Download models if missing
if not os.path.exists("yolov8m.pt"):
    print("Downloading YOLOv8 model...")
    YOLO("yolov8m.pt").export(format="onnx")
    
if not os.path.exists("yolov8m-pose.pt"):
    print("Downloading YOLOv8 Pose model...")
    YOLO("yolov8m-pose.pt").export(format="onnx")
    
class ChairTracker:
    def __init__(self):
        self.model = YOLO("yolov8m.pt")
        self.tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_cosine_distance=config.DEEPSORT_MAX_COS_DISTANCE
        )
        self.class_names = self.model.names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_iou(self, boxA, boxB):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    def track_chairs(self, video_path, output_path="chair_tracks.pkl"):
        """Track chairs in a video and save results to pickle file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return {}
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        chair_tracking_per_frame = {}
        unique_chair_ids = set()
        chair_id_to_bbox = {}
        frame_count = 0
        
        print(f"Tracking chairs in {video_path}...")
        pbar = tqdm(total=total_frames)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb, conf=config.CHAIR_CONFIDENCE)
            
            chair_detections_for_tracking = []
            chairs_in_frame = []
            
            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Only process chair detections
                    if cls == config.CHAIR_CLASS_ID and conf > config.CHAIR_CONFIDENCE:
                        w, h = x2 - x1, y2 - y1
                        
                        # Check IoU with existing tracked chairs
                        matched_id = None
                        for track_id, existing_bbox in chair_id_to_bbox.items():
                            iou = self.compute_iou((x1, y1, x2, y2), existing_bbox)
                            if iou > config.IOU_THRESHOLD:
                                matched_id = track_id
                                break
                        
                        if matched_id is not None:
                            # Update existing chair
                            chair_id_to_bbox[matched_id] = (x1, y1, x2, y2)
                            chairs_in_frame.append({
                                "id": matched_id,
                                "bbox": (x1, y1, x2, y2)
                            })
                        else:
                            # Add to tracker
                            chair_detections_for_tracking.append(((x1, y1, w, h), conf, 'chair'))
            
            # Update tracker
            tracks = self.tracker.update_tracks(chair_detections_for_tracking, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                
                # Update tracking data
                chair_id_to_bbox[track_id] = (x1, y1, x2, y2)
                unique_chair_ids.add(track_id)
                chairs_in_frame.append({
                    "id": track_id,
                    "bbox": (x1, y1, x2, y2)
                })
            
            # Save frame data
            chair_tracking_per_frame[frame_count] = chairs_in_frame
            frame_count += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        # Save results
        with open(output_path, "wb") as f:
            pickle.dump(chair_tracking_per_frame, f)
        
        print(f"✅ Tracked {len(unique_chair_ids)} unique chairs over {frame_count} frames")
        return chair_tracking_per_frame