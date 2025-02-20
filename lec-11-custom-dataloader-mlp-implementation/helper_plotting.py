import os
import numpy as np
import matplotlib.pyplot as plt

def plot_training_loss(minibatch_loss_list, 
                        num_epochs, iter_per_epoch,
                        results_dir=None, 
                        averaging_iterations=100):
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_loss_list)),
             minibatch_loss_list, label='Minibatch Loss')

    # Adjust y-axis limit if sufficient iterations have passed
    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([0, np.max(minibatch_loss_list[1000:]) * 1.5])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    # Plot running average of the loss
    running_avg = np.convolve(minibatch_loss_list,
                              np.ones(averaging_iterations) / averaging_iterations,
                              mode='valid')
    ax1.plot(running_avg, label='Running Average')
    ax1.legend()

    # Set a secondary x-axis for epochs
    ax2 = ax1.twiny()
    epoch_labels = list(range(num_epochs + 1))
    epoch_positions = [e * iter_per_epoch for e in epoch_labels]
    ax2.set_xticks(epoch_positions[::10])
    ax2.set_xticklabels(epoch_labels[::10])
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
        plt.savefig(image_path)
    else:
        plt.show()

def plot_accuracy(train_acc_list, valid_acc_list, results_dir=None):
    """
    Plots the training and validation accuracies over epochs.

    Args:
        train_acc_list (list or array): Training accuracies per epoch.
        valid_acc_list (list or array): Validation accuracies per epoch.
        results_dir (str, optional): Directory to save the plot PDF. If None, the plot is not saved.
    """
    num_epochs = len(train_acc_list)
    plt.figure()
    plt.plot(np.arange(1, num_epochs + 1), train_acc_list, label='Training Accuracy')
    plt.plot(np.arange(1, num_epochs + 1), valid_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_acc_training_validation.pdf')
        plt.savefig(image_path)
    else:
        plt.show()
