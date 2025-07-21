class Config:
    # YOLO model parameters
    CHAIR_CLASS_ID = 56
    CHAIR_CONFIDENCE = 0.6
    PERSON_CONFIDENCE = 0.3
    
    # Tracking parameters
    IOU_THRESHOLD = 0.7
    DEEPSORT_MAX_AGE = 70
    DEEPSORT_N_INIT = 5
    DEEPSORT_MAX_COS_DISTANCE = 0.4
    
    # Pose analysis parameters
    HIP_INSIDE_THRESHOLD = 0.1
    SEATED_LEG_ORDER_THRESHOLD = 100
    TYPING_POSTURE_THRESHOLD = 40

config = Config()