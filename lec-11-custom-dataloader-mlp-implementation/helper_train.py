import time
import torch
from helper_evaluation import compute_accuracy

def train_model(model, num_epochs, train_loader, valid_loader, test_loader, optimizer, device):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass and compute loss
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            
            # Backpropagation and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            minibatch_loss_list.append(loss.item())
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}')
        
        # Evaluate on training and validation sets
        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | Train: {train_acc:.2f}% | Validation: {valid_acc:.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())
        
        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')
    
    total_time = (time.time() - start_time) / 60
    print(f'Total Training Time: {total_time:.2f} min')
    
    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return minibatch_loss_list, train_acc_list, valid_acc_list