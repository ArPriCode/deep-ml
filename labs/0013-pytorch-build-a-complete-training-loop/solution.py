# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    # 1. Create optimizer: optim.Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 2. Create loss function: nn.CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    
    history = []
    num_train_samples = X_train.shape[0]
    
    # 3. For each epoch
    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        
        # a. Shuffle training data
        indices = torch.randperm(num_train_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        running_train_loss = 0.0
        num_batches = 0
        
        # b. Loop over mini-batches
        for i in range(0, num_train_samples, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()       # Clear gradients
            outputs = model(X_batch)    # Forward pass
            loss = criterion(outputs, y_batch) # Compute loss
            loss.backward()             # Backward pass
            optimizer.step()            # Update weights
            
            running_train_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = running_train_loss / num_batches
        
        # c. Compute validation accuracy (and validation loss)
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            predictions = val_outputs.argmax(dim=1)
            val_accuracy = (predictions == y_val).float().mean().item()
            
        # d. Append metrics to history
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
    # 4. Return history
    return history