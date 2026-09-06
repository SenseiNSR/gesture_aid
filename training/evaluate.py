import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.sign_language_model import SignLanguageLSTM
from training.train import load_data

def evaluate():
    print("Loading data...")
    X, y, action_map = load_data()
    if X is None:
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = len(action_map)
    model = SignLanguageLSTM(
        input_size=config.FEATURE_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=num_classes
    ).to(device)
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Model not found at {config.MODEL_SAVE_PATH}")
        return
        
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        
    preds = preds.cpu().numpy()
    
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, average='weighted', zero_division=0)
    recall = recall_score(y, preds, average='weighted', zero_division=0)
    f1 = f1_score(y, preds, average='weighted', zero_division=0)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    
    plt.figure(figsize=(10, 8))
    labels = [action_map[i] for i in range(num_classes)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(config.BASE_DIR, 'outputs', 'graphs', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Model size
    model_size = os.path.getsize(config.MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"\nModel Size: {model_size:.2f} MB")

if __name__ == "__main__":
    evaluate()
