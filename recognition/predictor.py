import torch
import json
import os
import numpy as np
from models.sign_language_model import SignLanguageLSTM
import config

class SignPredictor:
    def __init__(self, model_path=config.MODEL_SAVE_PATH, labels_path=config.LABELS_SAVE_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load labels
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.actions = json.load(f)
        else:
            self.actions = config.ACTIONS
            
        self.num_classes = len(self.actions)
        
        # Load model
        self.model = SignLanguageLSTM(
            input_size=config.FEATURE_DIM, 
            hidden_size=config.HIDDEN_SIZE, 
            num_layers=config.NUM_LAYERS, 
            num_classes=self.num_classes
        ).to(self.device)
        
        self.model_loaded = False
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model_loaded = True
            print(f"Model loaded successfully from {model_path} onto {self.device}")
        else:
            print(f"Warning: Model not found at {model_path}. Please train the model first or use Demo Mode.")

    def predict(self, sequence):
        """
        Predicts the sign given a sequence of features.
        sequence shape should be (sequence_length, feature_dim)
        """
        if not self.model_loaded:
            if config.DEMO_MODE:
                # Provide mock predictions in demo mode
                import random
                if random.random() > 0.95:
                    return random.choice(self.actions), 0.85 + random.random() * 0.1
                return "Unknown", 0.0
            return "Unknown", 0.0

        if len(sequence) != config.SEQUENCE_LENGTH:
            return "Unknown", 0.0
            
        seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            res = self.model(seq_tensor)
            probs = torch.softmax(res, dim=1).squeeze().cpu().numpy()
            
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        return self.actions[pred_idx], float(confidence)
