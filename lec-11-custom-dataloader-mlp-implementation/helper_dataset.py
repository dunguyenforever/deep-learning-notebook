import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        features = np.array([row['x1'],row['x2']], dtype = np.float32) # columns from the csv
        class_label = int(row['class label']) # true label colums from the csv
        return features, class_label

    def __len__(self):
        return len(self.data)
    
def get_loaders(csv_file, 
                batch_size, 
                train_transforms=None, 
                test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    
    if test_transforms is None:
        test_transforms = transforms.ToTensor()
    
    dataset = MyDataset(csv_file)
    n_samples = len(dataset)

    # Create and shuffle the indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 85% train, 10% test, 5% validation
    train_count = int(n_samples * 0.85)
    test_count = int(n_samples * 0.10)
    val_count = n_samples - train_count - test_count

    train_idx = indices[:train_count]
    test_idx = indices[train_count:train_count+test_count]
    val_idx = indices[train_count+test_count:]

    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)
    val_set = Subset(dataset, val_idx)

    # Creat the dataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, valid_loader