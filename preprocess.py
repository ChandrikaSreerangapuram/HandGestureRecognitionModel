import os
import json
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
    """
    Extracts 21 hand landmarks per hand and 9 key pose landmarks (upper body).
    Returns a flattened numpy array.
    """
    # 21 Hand landmarks * 3 (x,y,z) = 63 per hand. Total 126.
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    # 9 Pose landmarks (upper body) * 3 (x,y,z) = 27.
    # Keypoints: 0(nose), 11(l-shoulder), 12(r-shoulder), 13(l-elbow), 14(r-elbow), 
    # 15(l-wrist), 16(r-wrist), 23(l-hip), 24(r-hip)
    pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    pose = np.array([[results.pose_landmarks.landmark[i].x, 
                      results.pose_landmarks.landmark[i].y, 
                      results.pose_landmarks.landmark[i].z] for i in pose_indices]).flatten() if results.pose_landmarks else np.zeros(27)
    
    return np.concatenate([lh, rh, pose])

def process_videos(json_path, video_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    for video_id, info in tqdm(data.items(), desc="Processing videos"):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        save_path = os.path.join(output_dir, f"{video_id}.npy")
        
        if not os.path.exists(video_path):
            continue
        
        if os.path.exists(save_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make Detections
            results = holistic.process(image)
            
            # Extract Landmarks
            landmarks = extract_landmarks(results)
            sequence.append(landmarks)
            
        cap.release()
        
        # Save sequence as numpy array
        np.save(save_path, np.array(sequence))
        
    holistic.close()

if __name__ == "__main__":
    JSON_PATH = os.path.join('archive', 'nslt_300.json')
    VIDEO_DIR = os.path.join('archive', 'videos')
    OUTPUT_DIR = os.path.join('archive', 'processed_300')
    
    process_videos(JSON_PATH, VIDEO_DIR, OUTPUT_DIR)
