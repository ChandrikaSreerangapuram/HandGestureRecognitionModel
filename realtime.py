import cv2
import mediapipe as mp
import numpy as np
import torch
import os
import pyttsx3
import datetime
import time
from dotenv import load_dotenv
from model_hybrid import HybridASLModel
from preprocess import extract_landmarks
from gemini_integration import GeminiRefiner

load_dotenv()

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def load_labels(label_path, num_classes=300):
    labels = {}
    try:
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_classes:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    idx, gloss = parts[0], parts[1]
                    labels[int(idx)] = gloss
                else:
                    # Fallback if tab is missing
                    labels[i] = line.strip()
    except Exception as e:
        print(f"Error loading labels: {e}")
    return labels

def realtime_inference():
    # --- Configuration ---
    NUM_CLASSES = 50
    CONFIDENCE_THRESHOLD = 0.2  # Higher threshold for cleaner predictions
    STABILITY_FRAMES = 5
    PAUSE_FRAMES = 35
    SEQUENCE_LENGTH = 60
    
    device = torch.device("cpu") # Use "cuda" if available
    LABELS = load_labels(os.path.join('archive', 'wlasl_class_list.txt'), NUM_CLASSES)
    
    # Load class_map if exists (maps original class indices to 0..N)
    class_map_path = os.path.join('checkpoints', 'class_map.json')
    reverse_class_map = None
    if os.path.exists(class_map_path):
        import json as json_lib
        with open(class_map_path, 'r') as f:
            class_map = json_lib.load(f)
        # reverse: model output index -> original class index
        reverse_class_map = {v: int(k) for k, v in class_map.items()}
    
    # Initialize Log
    with open('realtime_output.log', 'w', encoding='utf-8') as f:
        f.write(f"--- ASL Sentence System Started: {datetime.datetime.now()} ---\n")

    # Load model
    model = HybridASLModel(input_dim=459, hidden_dim=128, num_classes=NUM_CLASSES, num_layers=2, dropout=0.5)
    checkpoint_path = os.path.join('checkpoints', 'best_asl_model.pth')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using untrained model.")
    model.eval()
    
    # Gemini Refiner
    # Recommendation: Set GEMINI_API_KEY environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not found in environment or .env file.")
        print("Gemini refinement will not work. Set it in your .env file.")
        GEMINI_API_KEY = ""
    refiner = GeminiRefiner(GEMINI_API_KEY)
    
    # TTS
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    # --- Variables for Inference ---
    sequence = []
    sentence_glosses = []
    prediction_history = []
    
    current_stable_word = ""
    last_added_word = ""
    
    frames_since_last_hand = 0
    refining = False
    
    # --- Setup Webcam ---
    cap = None
    import platform
    backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            print(f"Webcam opened at index {idx}")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    
    print("Inference running... 'q' to quit, 'c' to clear, 'r' to manual refine.")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # MediaPipe Detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Check if hands are present
            hands_detected = results.left_hand_landmarks or results.right_hand_landmarks
            
            if hands_detected:
                frames_since_last_hand = 0
                landmarks = extract_landmarks(results)
                sequence.append(landmarks)
                sequence = sequence[-SEQUENCE_LENGTH:]
                
                # Prediction Logic
                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.array(sequence).astype(np.float32)
                    velocity = np.diff(input_data, axis=0, append=input_data[-1:])
                    acceleration = np.diff(velocity, axis=0, append=velocity[-1:])
                    features = np.concatenate([input_data, velocity, acceleration], axis=-1)

                    with torch.no_grad():
                        input_tensor = torch.tensor(features).unsqueeze(0).to(device)
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probs, 1)
                        
                    conf_val = confidence.item()
                    pred_idx = predicted_idx.item()
                    # Map model output back to original class index
                    if reverse_class_map:
                        original_idx = reverse_class_map.get(pred_idx, pred_idx)
                    else:
                        original_idx = pred_idx
                    gloss = LABELS.get(original_idx, "Unknown")
                    
                    # Log for debugging
                    if conf_val > 0.1: # Only log non-negligible predictions
                         with open('realtime_output.log', 'a', encoding='utf-8') as f:
                            f.write(f"[DEBUG] {gloss} ({conf_val:.2f})\n")

                    # Stability Logic
                    if conf_val > CONFIDENCE_THRESHOLD:
                        prediction_history.append(gloss)
                    else:
                        prediction_history.append("None")
                    
                    prediction_history = prediction_history[-STABILITY_FRAMES:]
                    
                    # If the last N frames are the same and not "None"
                    if len(prediction_history) == STABILITY_FRAMES and len(set(prediction_history)) == 1:
                        word = prediction_history[0]
                        if word != "None" and word != last_added_word:
                            current_stable_word = word
                            sentence_glosses.append(word)
                            last_added_word = word
                            print(f"Recognized: {word}")
                            try:
                                engine.say(word)
                                engine.runAndWait()
                            except: pass
            else:
                frames_since_last_hand += 1
                # If hands are missing, we clear the sequence buffer to start fresh for the next sign
                if frames_since_last_hand > 5:
                    sequence = []
                    prediction_history = []
                    last_added_word = "" # Allow same word if they re-sign it
                
                # Auto-refine if there was a pause after some words were collected
                if frames_since_last_hand == PAUSE_FRAMES and len(sentence_glosses) > 0:
                    print("Pause detected. Refining sentence...")
                    refining = True
            
            # Handle Refinement (Auto or Manual)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('r') or refining:
                if len(sentence_glosses) > 0:
                    # Filter out any existing "Refined:" strings
                    clean_glosses = [g for g in sentence_glosses if not g.startswith("Refined:")]
                    
                    if clean_glosses:
                        # Use the correct object (refiner) and method (refine_sentence)
                        final_sentence = refiner.refine_sentence(clean_glosses)
                        
                        print(f"Final Sentence: {final_sentence}")
                        
                        # Store the refined sentence
                        sentence_glosses = [f"Refined: {final_sentence}"]

                        try:
                            engine.say(final_sentence)
                            engine.runAndWait()
                        except:
                            pass
                
                # Reset the auto-refine flag
                refining = False
                
            if key == ord('c'):
                sentence_glosses = []
                current_stable_word = ""
                print("Sentence cleared.")

            # --- UI VISUALIZATION ---
            # Draw Landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # UI Overlay
            # Bottom bar for sentence
            cv2.rectangle(image, (0, 440), (640, 480), (245, 117, 16), -1)
            display_text = " ".join(sentence_glosses)
            if len(display_text) > 40: display_text = "..." + display_text[-37:]
            cv2.putText(image, display_text, (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Top bar for current status
            cv2.rectangle(image, (0, 0), (640, 40), (50, 50, 50), -1)
            status_text = f"Status: {'Signing' if hands_detected else 'Idle'} | Stable: {current_stable_word}"
            cv2.putText(image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('ASL to Sentence System', image)
            
            if key == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_inference()
