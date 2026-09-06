import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.sign_language_model import SignLanguageLSTM

def load_data():
    sequences, labels = [], []
    action_map = {}
    
    if not os.path.exists(config.DATASET_DIR) or len(os.listdir(config.DATASET_DIR)) == 0:
        print(f"No data found in {config.DATASET_DIR}. Please run collect_data.py first.")
        return None, None, None
        
    actions = sorted([d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))])
    
    for label, action in enumerate(actions):
        action_map[label] = action
        action_path = os.path.join(config.DATASET_DIR, action)
        
        for seq_file in os.listdir(action_path):
            if seq_file.endswith('.npy'):
                res = np.load(os.path.join(action_path, seq_file))
                # Ensure sequence length is correct
                if res.shape == (config.SEQUENCE_LENGTH, config.FEATURE_DIM):
                    sequences.append(res)
                    labels.append(label)
                    
    return np.array(sequences), np.array(labels), action_map

def train():
    print("Loading data...")
    X, y, action_map = load_data()
    if X is None:
        return
        
    print(f"Loaded {len(X)} sequences across {len(action_map)} classes.")
    
    os.makedirs(os.path.dirname(config.LABELS_SAVE_PATH), exist_ok=True)
    # Save labels mapping
    with open(config.LABELS_SAVE_PATH, 'w') as f:
        json.dump([action_map[i] for i in range(len(action_map))], f)
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoaders
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageLSTM(
        input_size=config.FEATURE_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=len(action_map)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Starting training on {device}...")
    
    best_acc = 0.0
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seq.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for seq, labels in test_loader:
                seq, labels = seq.to(device), labels.to(device)
                outputs = model(seq)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * seq.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(test_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("  --> Saved best model")
            
    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
