# This script is part of the scene classification BSc work by Veera Kallio
# Here we train the pretrained VGG model for our data
# Tutorial used: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import torch
import torchvision
import pandas as pd
import numpy as np
import time
import copy
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            loss_acc[phase+"_loss"].append(epoch_loss)
            loss_acc[phase+"_acc"].append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = {"airport":0, "bus":1, "metro":2, "metro_station":3, "park":4 , "public_square":5,
          "shopping_mall":6, "street_pedestrian":7, "street_traffic":8, "tram":9}

image_datasets = {"train":[], "val":[]}

# Get the data
# Read the information from the csv-file
names = pd.read_csv("/lustre/audio-visual-scenes-2021/development/evaluation_setup/"
                    "fold1_train.csv", sep="\t")

for i, name in names.iterrows():
    # Get the video
    current = names.loc[i, ["filename_video", "scene_label"]]
    filename = current["filename_video"]
    # Get 3 frames per video
    for point in [0, 0.3, 0.6, 0.9]:
        frame, _, _ = torchvision.io.read_video("/lustre/audio-visual-scenes-2021/deve"
                                                "lopment/"+filename, start_pts=point,
                                                end_pts=point, pts_unit="sec")

        if frame.shape[0] != 0:
            # Resize and normalize images
            frame = frame.permute(0, 3, 1, 2).float()
            frame = frame[:][:]/255
            frame = F.interpolate(frame, size=(256,512))
            frame = torch.squeeze(frame)
            transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            frame = transform(frame)

            # Get the label
            label = current["scene_label"]
            # Split to train and validation according to city
            if filename.startswith(f"video/{label}-helsinki") or \
               filename.startswith(f"video/{label}-lisbon"):
                phase = "val"
            else:
                phase = "train"
            # Add to image_datasets
            image_datasets[phase].append([frame, classes[label]])


num_classes = 10
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=1) 
              for x in ['train', 'val']}

# Keep track of losses and accuracies
loss_acc = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

# Load pretrained VGG model
model = torchvision.models.vgg16_bn(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[6].out_features = 10
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=50)
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
