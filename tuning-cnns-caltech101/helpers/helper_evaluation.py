import os
import numpy as np
import random
import torch
from itertools import product


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """Sets PyTorch to deterministic mode for reproducibility."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # For PyTorch 1.8 and later
    torch.use_deterministic_algorithms(True)


def compute_accuracy(model, data_loader, device):
    """Computes classification accuracy of a given model."""
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_confusion_matrix(model, data_loader, device):
    """Computes a confusion matrix for a multi-class classification model."""
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            # Keep targets on CPU for convenient numpy usage
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.cpu())
            all_predictions.extend(predicted_labels.cpu())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Identify all class labels that appear in either predictions or targets
    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    # Handle edge case if there's only one label
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    
    n_labels = class_labels.shape[0]

    # Count occurrences of (target, prediction) pairs
    pairs = list(zip(all_targets, all_predictions))
    mat_vals = []
    for combi in product(class_labels, repeat=2):
        mat_vals.append(pairs.count(combi))

    confusion_mat = np.asarray(mat_vals).reshape(n_labels, n_labels)
    return confusion_mat
