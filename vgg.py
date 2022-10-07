# This script is part of the scene classification BSc work by Veera Kallio
# Here we train the pretrained VGG model for our data

import torch
import torchvision
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pickle
from read_data import read_data
from train_model import train_model

### Parameters to modify ###
frames = [0, 0.3, 0.6, 0.9]  # which frames to take from the video
image_size = (256,512)  # size to which we want to resize the images
batch_size = 32 
lr = 0.0001  # learning rate
lr_gamma = 0.1  # learning rate decay factor
step_size = 10  # reduce learning rate every step_size epochs
num_epochs = 50 
############################
num_classes = 10

# Get data
image_datasets = read_data(frames, image_size)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=1) 
              for x in ['train', 'val']}

# Load pretrained VGG model
model = torchvision.models.vgg16_bn(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
# Change the final output size
model.classifier[6].out_features = num_classes

# Loss function
criterion = torch.nn.CrossEntropyLoss()
# Different optimizers can be tested
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9, nesterov=True)
# Decay LR by a factor of gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)

# Train the model
model, loss_acc = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, 
                            exp_lr_scheduler, num_epochs)
# Save the model
pickle.dump(model, open("trained_vgg_model.sav", "wb"))

# Plot learning curves
plt.plot(loss_acc["train_loss"])
plt.plot(loss_acc["val_loss"])
plt.title("VGG loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["training", "validation"])
plt.savefig("images/vloss.png")
plt.clf()
plt.cla()

plt.plot(loss_acc["train_acc"])
plt.plot(loss_acc["val_acc"])
plt.title("VGG accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["training", "validation"])
plt.savefig("images/vacc.png")
