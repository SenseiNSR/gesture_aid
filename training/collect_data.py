import cv2
import numpy as np
import os
import time
import sys

# Add parent dir to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from recognition.hand_detector import HandDetector
from recognition.feature_extractor import extract_features

def collect_data():
    action = input(f"Enter the action you want to record (e.g., Hello, Thank you): ")
    if action not in config.ACTIONS:
        print(f"Warning: '{action}' is not in the default config.ACTIONS list. It will still be recorded.")
    
    try:
        num_sequences = int(input("How many sequences to record? (e.g. 30): "))
    except ValueError:
        print("Invalid number. Using 30.")
        num_sequences = 30
        
    action_path = os.path.join(config.DATASET_DIR, action)
    os.makedirs(action_path, exist_ok=True)
    
    # Find next available sequence number
    existing_seqs = [int(f.split('.')[0]) for f in os.listdir(action_path) if f.endswith('.npy')]
    start_seq_num = max(existing_seqs) + 1 if existing_seqs else 0
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    print("Camera opening... Get ready!")
    time.sleep(2)
    
    for sequence_num in range(start_seq_num, start_seq_num + num_sequences):
        # Wait between sequences to allow user to reset position
        print(f"\n--- Get ready for sequence {sequence_num+1} ---")
        time.sleep(1)
        
        sequence_data = []
        for frame_num in range(config.SEQUENCE_LENGTH):
            success, img = cap.read()
            if not success:
                break
                
            img = cv2.flip(img, 1)
            img, results = detector.find_hands(img)
            
            # Display info
            cv2.putText(img, f"Collecting: {action}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Seq: {sequence_num+1}/{start_seq_num + num_sequences} Frame: {frame_num+1}/{config.SEQUENCE_LENGTH}", 
                       (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow("Data Collection", img)
            
            # Extract and store features
            features = extract_features(results)
            sequence_data.append(features)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Collection interrupted.")
                return
                
        # Save sequence
        npy_path = os.path.join(action_path, f"{sequence_num}.npy")
        np.save(npy_path, np.array(sequence_data))
        print(f"Saved {npy_path}")
        
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")

if __name__ == "__main__":
    collect_data()
