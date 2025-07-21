import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from .config import config

class PoseAnalyzer:
    def __init__(self):
        self.pose_model = YOLO("yolov8m-pose.pt")
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    
    def compute_iou(self, boxA, boxB):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    def is_hip_inside_chair_box(self, person_keypoints, chair_boxes):
        """Check if either hip is inside any chair bounding box"""
        left_hip = person_keypoints[11]  # index 11 is left hip
        right_hip = person_keypoints[12]  # index 12 is right hip
        
        for chair in chair_boxes:
            x1, y1, x2, y2 = chair
            for hip in [left_hip, right_hip]:
                if hip is None:
                    continue
                hip_x, hip_y = hip[0], hip[1]
                if x1 <= hip_x <= x2 and y1 <= hip_y <= y2:
                    return True
        return False
    
    def get_best_chair_idx(self, person_box, chair_boxes):
        """Find best matching chair using IoU"""
        best_iou = 0
        best_idx = None
        for idx, chair in enumerate(chair_boxes):
            iou_val = self.compute_iou(person_box, chair)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx
        return best_idx, best_iou
    
    def is_seated_pose(self, person_keypoints, person_box, chair_boxes, verbose=False):
        """Determine if a person is in a seated position"""
        # Keypoint indices
        LEFT_HIP, RIGHT_HIP = 11, 12
        LEFT_KNEE, RIGHT_KNEE = 13, 14
        LEFT_ANKLE, RIGHT_ANKLE = 15, 16
        LEFT_ELBOW, RIGHT_ELBOW = 7, 8
        LEFT_WRIST, RIGHT_WRIST = 9, 10
        
        # Helper function to get coordinate
        def coord(idx):
            return person_keypoints[idx] if person_keypoints[idx] is not None else None
        
        # Check leg position
        def relaxed_leg_order(hip, knee, ankle):
            if hip is None or knee is None:
                return False
            if ankle is not None:
                return hip[1] < knee[1] and knee[1] <= ankle[1] + config.SEATED_LEG_ORDER_THRESHOLD
            return hip[1] < knee[1]
        
        cond1 = any([
            relaxed_leg_order(coord(LEFT_HIP), coord(LEFT_KNEE), coord(LEFT_ANKLE)),
            relaxed_leg_order(coord(RIGHT_HIP), coord(RIGHT_KNEE), coord(RIGHT_ANKLE))
        ])
        
        # Check hip position relative to chairs
        hip_coords = [coord(LEFT_HIP), coord(RIGHT_HIP)]
        hip_x_vals = [h[0] for h in hip_coords if h is not None]
        
        # Find chairs that align with hips horizontally
        x_matched_chairs = []
        for chair in chair_boxes:
            x1, _, x2, _ = chair
            for hip_x in hip_x_vals:
                if x1 <= hip_x <= x2:
                    x_matched_chairs.append(chair)
                    break
        
        cond2 = len(x_matched_chairs) > 0
        
        # Check if knees are within chair vertical bounds
        knee_y_vals = [k[1] for k in [coord(LEFT_KNEE), coord(RIGHT_KNEE)] if k is not None]
        cond3 = False
        
        for chair in x_matched_chairs:
            _, y1, _, y2 = chair
            for knee_y in knee_y_vals:
                if y1 <= knee_y <= y2:
                    cond3 = True
                    break
            if cond3:
                break
        
        # Check for typing posture (elbows and wrists at similar height)
        elbow_y_vals = [e[1] for e in [coord(LEFT_ELBOW), coord(RIGHT_ELBOW)] if e is not None]
        wrist_y_vals = [w[1] for w in [coord(LEFT_WRIST), coord(RIGHT_WRIST)] if w is not None]
        
        fallback_typing = False
        if elbow_y_vals and wrist_y_vals:
            avg_elbow_y = sum(elbow_y_vals) / len(elbow_y_vals)
            avg_wrist_y = sum(wrist_y_vals) / len(wrist_y_vals)
            if abs(avg_elbow_y - avg_wrist_y) < config.TYPING_POSTURE_THRESHOLD:
                fallback_typing = True
        
        # Final determination
        seated = (cond1 and cond2 and cond3) or fallback_typing
        return seated
    
    def analyze_frame(self, image_path, frame_number, chair_tracks, verbose=False):
        """Analyze a single frame for seated positions"""
        img = Image.open(image_path).convert("RGB")
        chairs_this_frame = chair_tracks.get(frame_number, [])
        chair_boxes = [c['bbox'] for c in chairs_this_frame]
        chair_ids = [c['id'] for c in chairs_this_frame]
        
        # Run pose detection
        pose_results = self.pose_model(img)
        keypoints_all = pose_results[0].keypoints.xy.cpu().numpy()
        person_boxes_all = pose_results[0].boxes.xyxy.cpu().numpy()
        
        occupied_chair_ids = set()
        seated_count = 0
        
        for keypoints, person_box in zip(keypoints_all, person_boxes_all):
            # Convert keypoints to list of (x, y) or None
            kpts = []
            for i in range(17):  # 17 keypoints
                if keypoints[i][0] == 0 and keypoints[i][1] == 0:
                    kpts.append(None)
                else:
                    kpts.append((float(keypoints[i][0]), float(keypoints[i][1])))
            
            # Check seated pose
            seated = self.is_seated_pose(kpts, person_box, chair_boxes, verbose)
            
            if seated:
                seated_count += 1
                best_idx, best_iou = self.get_best_chair_idx(person_box, chair_boxes)
                
                # Check if hips are inside chair as fallback
                hip_inside = self.is_hip_inside_chair_box(kpts, chair_boxes)
                
                if best_idx is not None and (best_iou > config.HIP_INSIDE_THRESHOLD or hip_inside):
                    occupied_chair_ids.add(chair_ids[best_idx])
        
        return len(chair_boxes), seated_count, sorted(occupied_chair_ids)