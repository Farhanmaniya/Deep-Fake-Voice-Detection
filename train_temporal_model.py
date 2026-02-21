"""
train_temporal_model.py
=======================
Training script for the secondary LSTM model using the FoR dataset.
Upgraded with higher capacity for better pattern recognition.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tqdm import tqdm
import random

# Configuration
DATA_DIR = r"D:\HACKATHONS\Deep_Fake_Detection\models\archive\for-2sec\for-2seconds"
SAVE_PATH = r"D:\HACKATHONS\Deep_Fake_Detection\models\temporal_lstm.pth"
SAMPLE_RATE = 16000
N_MELS = 64
MAX_FRAMES = 128 
BATCH_SIZE = 64 # Increased for stability
EPOCHS = 30     # Increased for better convergence
LEARNING_RATE = 0.0003 # Slower for deeper model

class FoRDataset(Dataset):
    def __init__(self, split="training"):
        self.files = []
        self.labels = []
        
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = os.path.join(DATA_DIR, split, class_name)
            if not os.path.exists(class_dir):
                continue
            for f in os.listdir(class_dir):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(class_dir, f))
                    self.labels.append(float(label))
                    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=2.0)
            mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=1.0)
            mel_db = np.clip(mel_db, -80, 0)
            mel_db = (mel_db + 40.0) / 40.0 
            
            if mel_db.shape[1] < MAX_FRAMES:
                pad_width = MAX_FRAMES - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_db = mel_db[:, :MAX_FRAMES]
            
            return torch.FloatTensor(mel_db.T), torch.FloatTensor([label])
        except:
            return torch.zeros((MAX_FRAMES, N_MELS)), torch.FloatTensor([label])

class TemporalLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=256, num_layers=3): # Capacity: 256
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} (Epochs: {EPOCHS}, Hidden: 256)")
    
    train_dataset = FoRDataset("training")
    print(f"Total training files: {len(train_dataset.files)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0 # Stable for CPU
    )
    
    model = TemporalLSTM(input_size=N_MELS).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=True)
        for features, labels in loop:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item(), acc=f"{(100*correct/total):.1f}%")
            
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()
