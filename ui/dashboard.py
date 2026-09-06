import customtkinter as ctk
from PIL import Image
import cv2

class SignLanguageDashboard(ctk.CTk):
    def __init__(self, start_cam_cb, stop_cam_cb, speak_cb, clear_cb, delete_cb, save_cb):
        super().__init__()
        
        self.title("Real-Time Sign Language Translator")
        self.geometry("1000x600")
        
        # Callbacks
        self.start_cam_cb = start_cam_cb
        self.stop_cam_cb = stop_cam_cb
        self.speak_cb = speak_cb
        self.clear_cb = clear_cb
        self.delete_cb = delete_cb
        self.save_cb = save_cb
        
        # UI Setup
        self.setup_ui()
        
    def setup_ui(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Left Frame: Camera
        self.cam_frame = ctk.CTkFrame(self)
        self.cam_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.vid_label = ctk.CTkLabel(self.cam_frame, text="Camera Feed Off")
        self.vid_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Right Frame: Stats & Controls
        self.stats_frame = ctk.CTkFrame(self)
        self.stats_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(self.stats_frame, text="RECOGNITION", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        self.lbl_prediction = ctk.CTkLabel(self.stats_frame, text="Detected: None", font=ctk.CTkFont(size=16))
        self.lbl_prediction.pack(pady=5, anchor="w", padx=20)
        
        self.lbl_confidence = ctk.CTkLabel(self.stats_frame, text="Confidence: 0%", font=ctk.CTkFont(size=16))
        self.lbl_confidence.pack(pady=5, anchor="w", padx=20)
        
        self.lbl_fps = ctk.CTkLabel(self.stats_frame, text="FPS: 0", font=ctk.CTkFont(size=16))
        self.lbl_fps.pack(pady=5, anchor="w", padx=20)
        
        self.lbl_latency = ctk.CTkLabel(self.stats_frame, text="Latency: 0 ms", font=ctk.CTkFont(size=16))
        self.lbl_latency.pack(pady=5, anchor="w", padx=20)
        
        self.lbl_mode = ctk.CTkLabel(self.stats_frame, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color="red")
        self.lbl_mode.pack(pady=20)
        
        # Camera buttons
        self.btn_start = ctk.CTkButton(self.stats_frame, text="Start Camera", command=self.start_cam_cb)
        self.btn_start.pack(pady=10)
        
        self.btn_stop = ctk.CTkButton(self.stats_frame, text="Stop Camera", command=self.stop_cam_cb)
        self.btn_stop.pack(pady=10)
        
        # Bottom Frame: Text Output & Controls
        self.bottom_frame = ctk.CTkFrame(self, height=150)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.bottom_frame, text="TRANSLATED SENTENCE", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.lbl_sentence = ctk.CTkLabel(self.bottom_frame, text="", font=ctk.CTkFont(size=24))
        self.lbl_sentence.pack(pady=10)
        
        # Action Buttons
        self.actions_frame = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        self.actions_frame.pack(pady=10)
        
        ctk.CTkButton(self.actions_frame, text="DELETE", command=self.delete_cb, width=100).pack(side="left", padx=10)
        ctk.CTkButton(self.actions_frame, text="CLEAR", command=self.clear_cb, width=100).pack(side="left", padx=10)
        ctk.CTkButton(self.actions_frame, text="SPEAK", command=self.speak_cb, width=100).pack(side="left", padx=10)
        ctk.CTkButton(self.actions_frame, text="SAVE", command=self.save_cb, width=100).pack(side="left", padx=10)

    def update_frame(self, frame):
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
        self.vid_label.configure(image=imgtk, text="")
        self.vid_label.image = imgtk

    def update_stats(self, prediction, confidence, fps, latency, mode_text=""):
        self.lbl_prediction.configure(text=f"Detected: {prediction}")
        self.lbl_confidence.configure(text=f"Confidence: {int(confidence*100)}%")
        self.lbl_fps.configure(text=f"FPS: {fps}")
        self.lbl_latency.configure(text=f"Latency: {latency:.1f} ms")
        if mode_text:
            self.lbl_mode.configure(text=mode_text)
            
    def update_sentence(self, sentence):
        self.lbl_sentence.configure(text=sentence)
