# This script is part of the scene classification BSc work by Veera Kallio
# This is a function for reading the data

import pandas as pd
import torch
import torchvision
import torch.nn.functional as F

def read_data(frames=[0, 0.3, 0.6, 0.9], frame_size=(256,512)):
    """
    Load TAU Urban Audio-Visual Scenes 2021 dataset.

    Input parameters:
    - frames: list of times of the video between 0-1, where we want to take frames
    - frame_size: output video frame size

    Returns:
    - image_datasets: training and validation video frames and labels (0-9), 
    {'train': [Tensor, int], 'val': [Tensor, int]}

    """
    image_datasets = {"train":[], "val":[]}
    classes = {"airport":0, "bus":1, "metro":2, "metro_station":3, "park":4 , "public_square":5,
          "shopping_mall":6, "street_pedestrian":7, "street_traffic":8, "tram":9}

    # Read the dataset information from the csv-file
    names = pd.read_csv("/lustre/audio-visual-scenes-2021/development/evaluation_setup/"
                        "fold1_train.csv", sep="\t")

    for i, name in names.iterrows():
        # Get the video name
        current = names.loc[i, ["filename_video", "scene_label"]]
        filename = current["filename_video"]
        # Get the video frames we want to use
        for point in frames:
            frame, _, _ = torchvision.io.read_video("/lustre/audio-visual-scenes-2021/deve"
                                                    "lopment/"+filename, start_pts=point,
                                                    end_pts=point, pts_unit="sec")

            if frame.shape[0] != 0:
                # Resize and normalize images
                frame = frame.permute(0, 3, 1, 2).float()
                frame = frame[:][:]/255
                frame = F.interpolate(frame, frame_size)
                frame = torch.squeeze(frame)
                transform = torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])  # needed for pretrained torchvision models
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

    return image_datasets
                