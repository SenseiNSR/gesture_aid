import cv2
import time
import collections
from recognition.hand_detector import HandDetector
from recognition.feature_extractor import extract_features
from recognition.predictor import SignPredictor
from recognition.temporal_smoother import TemporalSmoother
from utils.fps import FPSMonitor
from utils.tts import TextToSpeech
from ui.dashboard import SignLanguageDashboard
import config

class SignLanguageApp:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.predictor = SignPredictor()
        self.smoother = TemporalSmoother()
        self.fps_monitor = FPSMonitor()
        self.tts = TextToSpeech()
        
        self.sequence = collections.deque(maxlen=config.SEQUENCE_LENGTH)
        self.sentence = []
        
        self.is_running = False
        self.cap = None
        
        self.ui = SignLanguageDashboard(
            start_cam_cb=self.start_camera,
            stop_cam_cb=self.stop_camera,
            speak_cb=self.speak_sentence,
            clear_cb=self.clear_sentence,
            delete_cb=self.delete_word,
            save_cb=self.save_sentence
        )
        
        self.mode_text = "DEMO MODE - NOT REAL MODEL PREDICTION" if config.DEMO_MODE else ""
        if not self.predictor.model_loaded and not config.DEMO_MODE:
            self.mode_text = "MODEL NOT FOUND - PLEASE TRAIN"

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.update_loop()
            
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.ui.vid_label.configure(image=None, text="Camera Feed Off")
        
    def speak_sentence(self):
        text = " ".join(self.sentence)
        if text:
            self.tts.speak(text)
            
    def clear_sentence(self):
        self.sentence = []
        self.ui.update_sentence("")
        
    def delete_word(self):
        if self.sentence:
            self.sentence.pop()
            self.ui.update_sentence(" ".join(self.sentence))
            
    def save_sentence(self):
        text = " ".join(self.sentence)
        if text:
            with open("saved_sentences.txt", "a") as f:
                f.write(text + "\n")
            print(f"Saved: {text}")

    def process_word(self, word):
        if word == "SPACE":
            pass # Just a separator
        elif word == "DELETE":
            self.delete_word()
        elif word == "CLEAR":
            self.clear_sentence()
        else:
            self.sentence.append(word)
            self.ui.update_sentence(" ".join(self.sentence))

    def update_loop(self):
        if self.is_running and self.cap.isOpened():
            start_time = time.time()
            
            success, img = self.cap.read()
            if success:
                img = cv2.flip(img, 1) # Mirror
                img, results = self.hand_detector.find_hands(img)
                
                prediction = "Unknown"
                confidence = 0.0
                
                if results.multi_hand_landmarks:
                    features = extract_features(results)
                    self.sequence.append(features)
                    
                    if len(self.sequence) == config.SEQUENCE_LENGTH:
                        prediction, confidence = self.predictor.predict(list(self.sequence))
                        
                        smoothed_pred = self.smoother.process(prediction, confidence)
                        if smoothed_pred:
                            self.process_word(smoothed_pred)
                
                fps = self.fps_monitor.update()
                latency = (time.time() - start_time) * 1000
                
                self.ui.update_stats(prediction, confidence, fps, latency, self.mode_text)
                self.ui.update_frame(img)
            
            self.ui.after(10, self.update_loop)

    def run(self):
        self.ui.mainloop()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()
