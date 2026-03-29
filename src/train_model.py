import torch
from torch.utils.data import Dataset, DataLoader # DataLoader isn't really gonna be used here actually but oh well
import pandas as pd
    
import torch.nn as nn
import torch.nn.functional as F


class GestureDataset(Dataset):
    def __init__(self, csv_file):
        # Load the entire CSV into memory using Pandas
        self.data = pd.read_csv(csv_file)
        
        # .iloc[:, 1:] means get all rows, and all columns EXCEPT the first one
        raw_x = self.data.iloc[:, 1:].values
        # .iloc[:, 0] means "Get all rows, but ONLY the first column"
        raw_y = self.data.iloc[:, 0].values
        
        # Converting raw arrays into pytorch sensors
        self.X = torch.tensor(raw_x, dtype=torch.float32)
        self.Y = torch.tensor(raw_y, dtype=torch.long)

    def __len__(self):
        # Pytotoorch needs to know exactly how many rows of data exist
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch uses this to grab one specific row during training
        return self.X[idx], self.Y[idx]

class GestureNetwork(nn.Module):
    def __init__(self):
        super(GestureNetwork, self).__init__()
        
        # Layer 1: The Input Layer
        # Takes in our 42 coordinates (21 X, 21 Y), outputs to 128 hidden neurons
        self.fc1 = nn.Linear(in_features=42, out_features=128)
        
        # Layer 2: The Hidden Layer
        # Takes the 128 neurons, narrows them down to 64
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        
        # Layer 3: The Output Layer
        # Takes the 64 neurons and outputs exactly 4 numbers (our 4 gestures)
        self.fc3 = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        # This defines exactly how the data flows through the layers
        
        # Pass through Layer 1, then apply ReLU activation
        x = F.relu(self.fc1(x))
        
        # Pass through Layer 2, then apply ReLU activation
        x = F.relu(self.fc2(x))
        
        # Pass through the final output layer
        x = self.fc3(x) # No ReLU here cuz this outputs the model's 3 confidence scores
        
        return x

import torch.optim as optim

if __name__ == "__main__":
    print("--- INITIALIZING TRAINING ENGINE ---")
    
    # 1. Load the Data
    dataset = GestureDataset('gesture_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Boot up the Motherboard
    brain = GestureNetwork()
    
    # 3. Hire the Grader (Loss) and the Mechanic (Optimizer)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(brain.parameters(), lr=0.001) 
    
    # 4. The Epoch Loop
    epochs = 50
    print("--- STARTING TRAINING ---")
    
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            
            # Forward Pass: AI guesses the gesture
            predictions = brain(batch_x)
            
            # Loss: We calculate how wrong the guess was
            loss = criterion(predictions, batch_y)
            
            # Backprop & Optimize: We fix the math weights
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      
            
        # Print the error score every 5 epochs so we can watch it learn
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Error Score (Loss): {loss.item():.4f}")

    # 5. Freeze the math and save the file
    torch.save(brain.state_dict(), 'gesture_model.pth')
    print("--- TRAINING COMPLETE: Model saved as 'gesture_model.pth' ---")