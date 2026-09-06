import collections
import config

class TemporalSmoother:
    def __init__(self, threshold=config.CONFIDENCE_THRESHOLD, window_size=config.SMOOTHING_WINDOW, cooldown=config.COOLDOWN_FRAMES):
        self.threshold = threshold
        self.predictions_window = collections.deque(maxlen=window_size)
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.last_prediction = None
        
    def process(self, prediction, confidence):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            
        if confidence < self.threshold:
            self.predictions_window.append("Unknown")
            return None
            
        self.predictions_window.append(prediction)
        
        # Check if the most common prediction in the window is not "Unknown"
        if len(self.predictions_window) == self.predictions_window.maxlen:
            most_common = max(set(self.predictions_window), key=self.predictions_window.count)
            count = self.predictions_window.count(most_common)
            
            # If the most common prediction occurs frequently enough
            if count >= self.predictions_window.maxlen - 1 and most_common != "Unknown":
                if self.cooldown_counter == 0 or most_common != self.last_prediction:
                    self.last_prediction = most_common
                    self.cooldown_counter = self.cooldown
                    self.predictions_window.clear() # Reset window after a confident prediction
                    return most_common
                    
        return None
