# This script is part of the scene classification BSc work by Veera Kallio
# This is a function for training a CNN model

import torch
import time
import copy

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs):
    """
    Train a PyTorch model.
    
    Input parameters:
    - dataloaders: training and validation images and labels in the form 
    {'train': torch DataLoader, 'val': torch DataLoader}
    - dataset_sizes: dataset sizes, {'train': int, 'val': int}
    - model: torch model
    - criterion: loss function
    - optimizer: model optimizer
    - scheduler: learning rate scheduler
    - num_epochs: number of training epochs, int

    Returns:
    - model: the trained torch model
    - loss_acc: losses and accuracies during training in the form
    {"train_loss": [float], "val_loss": [float], "train_acc": [float], "val_acc": [float]}

    """
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()
    # Keep track of losses and accuracies
    loss_acc = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # Results
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            loss_acc[phase+"_loss"].append(epoch_loss)
            loss_acc[phase+"_acc"].append(epoch_acc)

            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_acc
