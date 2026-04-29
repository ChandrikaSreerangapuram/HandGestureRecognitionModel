"""
Record your own sign language training data.
This will capture landmark sequences for signs YOU perform,
so the model learns YOUR hands and environment.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Simple gestures anyone can do (NO ASL knowledge needed!)
SIGNS = [
    "hello",       # Wave your hand
    "yes",         # Thumbs up
    "no",          # Wave finger side to side
    "stop",        # Open palm facing camera
    "thank you",   # Flat hand from chin forward
    "please",      # Rub chest in circle
    "help",        # Fist on open palm, lift up
    "good",        # Thumbs up
    "bad",         # Thumbs down
    "i love you",  # Pinky + index + thumb extended
]

NUM_SAMPLES_PER_SIGN = 20  # Record 20 samples of each sign
SEQUENCE_LENGTH = 40       # Frames per sample

OUTPUT_DIR = "custom_data"

def extract_landmarks(results):
    """Same extraction as preprocess.py"""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    
    pose_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    pose = np.array([[results.pose_landmarks.landmark[i].x, 
                      results.pose_landmarks.landmark[i].y, 
                      results.pose_landmarks.landmark[i].z] for i in pose_indices]).flatten() if results.pose_landmarks else np.zeros(27)
    
    return np.concatenate([lh, rh, pose])


def record_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for sign_idx, sign in enumerate(SIGNS):
            sign_dir = os.path.join(OUTPUT_DIR, sign)
            os.makedirs(sign_dir, exist_ok=True)
            
            # Check how many samples already exist
            existing = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
            
            for sample_num in range(existing, NUM_SAMPLES_PER_SIGN):
                # Show instruction
                print(f"\n[{sign_idx+1}/{len(SIGNS)}] Sign: '{sign.upper()}' — Sample {sample_num+1}/{NUM_SAMPLES_PER_SIGN}")
                print("Press SPACE to start recording, 'q' to quit, 's' to skip sign")
                
                # Wait for user to be ready
                waiting = True
                while waiting:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Show instruction on frame
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (0, 0), (640, 60), (50, 50, 50), -1)
                    cv2.putText(frame, f"Sign: '{sign.upper()}' [{sample_num+1}/{NUM_SAMPLES_PER_SIGN}]", 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "SPACE=Record | S=Skip | Q=Quit", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    cv2.imshow('Record Training Data', frame)
                    key = cv2.waitKey(10) & 0xFF
                    
                    if key == ord(' '):
                        waiting = False
                    elif key == ord('s'):
                        waiting = False
                        break
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Recording stopped.")
                        return
                
                if key == ord('s'):
                    print(f"Skipping '{sign}'")
                    break
                
                # Countdown
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (200, 180), (440, 300), (0, 0, 0), -1)
                    cv2.putText(frame, str(countdown), (280, 270), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                    cv2.imshow('Record Training Data', frame)
                    cv2.waitKey(1000)
                
                # Record sequence
                sequence = []
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    
                    landmarks = extract_landmarks(results)
                    sequence.append(landmarks)
                    
                    # Draw landmarks and progress
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                    # Progress bar
                    progress = int((frame_num + 1) / SEQUENCE_LENGTH * 600)
                    cv2.rectangle(frame, (20, 450), (620, 470), (50, 50, 50), -1)
                    cv2.rectangle(frame, (20, 450), (20 + progress, 470), (0, 255, 0), -1)
                    cv2.putText(frame, f"RECORDING '{sign.upper()}'", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    cv2.imshow('Record Training Data', frame)
                    cv2.waitKey(30)
                
                # Save
                save_path = os.path.join(sign_dir, f"sample_{sample_num:03d}.npy")
                np.save(save_path, np.array(sequence))
                print(f"  Saved: {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Recording complete! Data saved to '{OUTPUT_DIR}/'")
    print(f"  Signs: {len(SIGNS)}, Samples per sign: {NUM_SAMPLES_PER_SIGN}")
    print(f"  Next step: python3 train_custom.py")


if __name__ == "__main__":
    print("=" * 50)
    print("  ASL CUSTOM DATA RECORDER")
    print("=" * 50)
    print(f"\nGestures to record (just follow the description!):")
    print()
    descriptions = {
        "hello": "Wave your hand left-right",
        "yes": "Thumbs up (or nod fist up-down)",
        "no": "Wave index finger side to side",
        "stop": "Open palm facing camera, hold still",
        "thank you": "Flat hand touches chin, move forward",
        "please": "Open hand rubs chest in circle",
        "help": "Fist on open palm, lift both up",
        "good": "Thumbs up, move away from chin",
        "bad": "Thumbs down",
        "i love you": "Extend pinky + index + thumb (🤟)",
    }
    for i, sign in enumerate(SIGNS):
        desc = descriptions.get(sign, "Perform gesture")
        print(f"  {i+1}. {sign.upper():12s} → {desc}")
    
    print(f"\nSamples per gesture: {NUM_SAMPLES_PER_SIGN}")
    print(f"Total recordings: {len(SIGNS) * NUM_SAMPLES_PER_SIGN}")
    print(f"\nTips:")
    print("  - Just follow the description above")
    print("  - Vary speed slightly between samples")
    print("  - Keep hands in frame")
    print("  - Takes ~10 minutes total")
    print()
    record_data()
