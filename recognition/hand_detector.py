import cv2
import time
import mediapipe as mp

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Initialize the hand landmarker using the new Tasks API
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = HandLandmarker.create_from_options(options)

    def find_hands(self, img, draw=True):
        # The new API requires a timestamp for VIDEO mode
        timestamp_ms = int(time.time() * 1000)
        
        # Convert BGR to RGB and then to a MediaPipe Image
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        # Process the image
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        # We need to emulate the old results structure to stay compatible with feature_extractor.py
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class MockClassification:
            def __init__(self, label):
                self.label = label
                
        class MockClassificationList:
            def __init__(self, label):
                self.classification = [MockClassification(label)]

        class MockHandLandmarks:
            def __init__(self, lms):
                self.landmark = lms

        class MockResults:
            def __init__(self):
                self.multi_hand_landmarks = None
                self.multi_handedness = None

        mapped_results = MockResults()

        if detection_result.hand_landmarks:
            mapped_results.multi_hand_landmarks = []
            for hand_lms in detection_result.hand_landmarks:
                # hand_lms is a list of NormalizedLandmark objects
                mock_hand = [MockLandmark(lm.x, lm.y, lm.z) for lm in hand_lms]
                mapped_results.multi_hand_landmarks.append(MockHandLandmarks(mock_hand))
                
            mapped_results.multi_handedness = []
            if detection_result.handedness:
                for handedness_list in detection_result.handedness:
                    # handedness_list is a list of Category objects, we usually take the first one
                    mapped_results.multi_handedness.append(MockClassificationList(handedness_list[0].category_name))

            if draw:
                h, w, c = img.shape
                for hand_lms in detection_result.hand_landmarks:
                    for i, lm in enumerate(hand_lms):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                    
                    # Optional: We could draw lines here, but drawing points is enough for visual feedback
                    
        return img, mapped_results
