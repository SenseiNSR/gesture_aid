import pyttsx3
import threading

class TextToSpeech:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150) # Speed of speech
            self.is_speaking = False
            self.available = True
        except Exception as e:
            print(f"Failed to initialize TTS: {e}")
            self.available = False

    def _speak_thread(self, text):
        self.is_speaking = True
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False

    def speak(self, text):
        if not self.available:
            return
        if not self.is_speaking and text.strip():
            # Run in a separate thread so it doesn't block the UI/webcam
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
