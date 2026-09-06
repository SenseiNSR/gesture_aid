import numpy as np

def extract_features(results):
    """
    Extracts and normalizes hand landmarks from MediaPipe results.
    Returns a flat array of 126 features (21 landmarks * 3 coords * 2 hands).
    """
    # Initialize empty arrays for left and right hands
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract landmarks for the hand
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            
            # Determine if it's left or right hand using handedness
            if results.multi_handedness:
                # MediaPipe handedness is mirrored by default, but we'll use the label provided
                handedness_label = results.multi_handedness[i].classification[0].label
                if handedness_label == 'Left':
                    lh = landmarks
                else:
                    rh = landmarks
            else:
                # Fallback if handedness is not available
                if i == 0:
                    lh = landmarks
                else:
                    rh = landmarks
                    
    # Concatenate left and right hand features
    # Total dimension: 63 + 63 = 126
    return np.concatenate([lh, rh])
