import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "sequences")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "sign_language_model.pth")
LABELS_SAVE_PATH = os.path.join(MODEL_DIR, "labels.json")

# Model hyperparameters
SEQUENCE_LENGTH = 30
# 21 landmarks * (x,y,z) * 2 hands = 126
FEATURE_DIM = 126 
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Recognition
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5 # Consecutive frames needed for prediction
COOLDOWN_FRAMES = 20

# Demo Mode Settings
DEMO_MODE = False

# Make sure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ACTIONS = [
    "Hello", "Thank you", "Yes", "No", "Please", "Help",
    "Sorry", "Good", "Bad", "Stop", "Water", "Food",
    "I", "You", "Love", "Welcome", "SPACE", "DELETE", "CLEAR"
]
