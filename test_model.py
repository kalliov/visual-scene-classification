# This script is part of the scene classification BSc work by Veera Kallio
# Here we evaluate the trained CNN model

import torch
import torchvision
import pandas as pd
import numpy as np
import time
import copy
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn.functional as F
import pickle


classes = {"airport":0, "bus":1, "metro":2, "metro_station":3, "park":4 , "public_square":5,
          "shopping_mall":6, "street_pedestrian":7, "street_traffic":8, "tram":9}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the saved model
model = pickle.load(open("trained_resnet_model.sav", "rb"))
model.eval()
criterion = torch.nn.CrossEntropyLoss()

running_loss = np.zeros([10])
running_corrects = np.zeros([10])
sizes = np.zeros([10])


# Read the information from the csv-file
names = pd.read_csv("/lustre/audio-visual-scenes-2021/development/evaluation_setup/"
                    "fold1_evaluate.csv", sep="\t")

for i, name in names.iterrows():
    # Get the video
    current = names.loc[i, ["filename_video", "scene_label"]]
    frames = []
    # Use 10 frames per video in testing
    for point in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        frame, _, _ = torchvision.io.read_video("/lustre/audio-visual-scenes-2021/deve"
                                                "lopment/"+current["filename_video"],
                                        start_pts=point, end_pts=point, pts_unit="sec")
        frame = frame.to(device)
        if frame.shape[0] != 0:
            # Resize and normalize like in training phase
            frame = frame.permute(0, 3, 1, 2).float()
            frame = frame[:][:]/255
            frame = F.interpolate(frame, size=(256,512))
            transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            frame = transform(torch.squeeze(frame))
            frames.append(frame)
    
    if frames != []:
        batch = torch.stack(frames, 0)
        
        # Get the label
        label = classes[current["scene_label"]]
        label = torch.tensor([label])
        label = label.to(device)
    
        # Evaluation    
        output = model(batch)
        output = torch.mean(output, 0, keepdim=True)
        _, pred = torch.max(output, 1)
        loss = criterion(output, label)
        print(label, pred)

        # Statistics
        running_loss[label] += loss.item()
        running_corrects[label] += (pred == label)
        sizes[label] += 1

# Results
loss = running_loss / sizes
acc = running_corrects / sizes
print()
print("Overall loss: {:.4f}, accuracy: {:.4f}".format(np.mean(loss), np.mean(acc)))
print()
print("{:20} loss\t accuracy".format("class"))
for label, i in classes.items():
    print("{:20} {:.4f}\t {:.4f}".format(label, loss[i], acc[i]))

