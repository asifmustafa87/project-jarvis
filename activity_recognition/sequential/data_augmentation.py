import os
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2 as cv
from PIL import Image
import json
import shutil
import matplotlib.pyplot as plt



ACTIVITIES = {
    "take leg": 0,
    "assemble leg": 1,
    "grab drill": 2,
    "use drill": 3,
    "drop drill": 4,
    "take screw driver": 5,
    "use screw driver": 6,
    "drop screw driver": 7
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

root_dir = os.path.join("activity recognition", "mock data", "actions")
augmented_dir = os.path.join("activity recognition", "mock data", "augmented_actions")
counter = 0 # to remain order
rounds = 3
for i in range(rounds):
    print("round:", i)
    for activity in os.listdir(root_dir):
        folders = [folder for folder in os.listdir(os.path.join(root_dir, activity))\
               if os.path.isdir(os.path.join(root_dir, activity, folder))]
        for folder in folders:
            path = os.path.join(root_dir, activity, folder)
            if len(os.listdir(path)) < 3:
                print("skipped smaller than 3 sequence")
                continue
            for img_name in os.listdir(path):
                img = cv.imread(os.path.join(path, img_name))
                augmented = transform(img)
                plt.imsave(os.path.join(augmented_dir, activity, "aug_" + str(counter) + ".jpg"), augmented.permute(1, 2, 0).numpy())
                counter += 1

for activity in os.listdir(augmented_dir):
    # if activity == "assemble leg":
    #     continue
    folder_count = 100
    c = -1
    os.mkdir(os.path.join(augmented_dir, activity, str(folder_count)))
    for img in os.listdir(os.path.join(augmented_dir, activity)):
        path = os.path.join(augmented_dir, activity, img)
        if c < 2:
            c += 1
        elif c == 2:
            c = 0
            folder_count += 1
            os.mkdir(os.path.join(augmented_dir, activity, str(folder_count)))

        if os.path.isdir(path):
            c -= 1
            continue
        new_path = os.path.join(augmented_dir, activity, str(folder_count), img)
        shutil.move(path, new_path)


            


