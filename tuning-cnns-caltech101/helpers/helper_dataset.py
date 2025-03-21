import torch
from torch.utils.data import sampler
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

class UnNormalize(object):
    def __init__(self,mean,std):
        self.mean = mean 
        self.std = std 
    
    def __call__(self, tensor):
        """
        Parameters:
        ------------
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        
        Returns:
        ------------
        Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t @= s
            t += m
        return tensor

def get_dataloaders_caltech101(batch_size, num_workers=0,
                                validation_fraction=None,
                                train_transforms=None,
                                test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
    root_dir = '../data'
    train_dataset = datasets.Caltech101(root=root_dir, 
                                        download=True, 
                                        target_type='category',
                                        transform=train_transforms)

    valid_dataset = datasets.Caltech101(root=root_dir,
                                        target_type='category',
                                        transform=test_transforms)

    test_dataset = datasets.Caltech101(root=root_dir,
                                        target_type='category',
                                        transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * len(train_dataset))
        
        indices = torch.randperm(len(train_dataset))
        train_indices = indices[:-num]
        valid_indices = indices[-num:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True,
                                sampler=train_sampler)

    else:
        train_loader = DataLoader(datatset=train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True,
                                shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else: 
        return train_loader, valid_loader, test_loader


    

 


