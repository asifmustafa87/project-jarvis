# Validation code of transformer for activity recognition
# Confusion Matrix for Predicted classes
# Provide reference to this code (hamas.khan@tum.de) if using it for your project
# Part of the Sandbox_Hamas repository


import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import shutil
from sklearn.metrics import confusion_matrix

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

class Transformer(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads, num_layers):
        super(Transformer, self).__init__()

        self.embed_dim = embed_dim

        self.embedding = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, _, _, _ = x.shape

        x = self.embedding(x)
        x = x.permute(2, 3, 0, 1)

        seq_length = x.shape[2]
        x = x.reshape(-1, seq_length, self.embed_dim)

        x = self.transformer(x)

        x = x.mean(dim=0)
        x = self.fc(x)

        return x

# Replace these configurations according to the loaded model. Config can be found in /Hyperparameter Tuning/...
# Hyperparameters
num_classes = 8
patch_size = 32
embed_dim = 256
num_heads = 2
num_layers = 1

learning_rate = 1e-4
num_epochs = 100
dropout_prob = 0.5
l2_lambda = 1e-5

batch_size = 1

# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([
    Resize((patch_size, patch_size)),
    ToTensor(),
])

# Loading the Test Dataset
test_data_dir = "Test Data By Class Renamed"
test_dataset = CustomDataset(test_data_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Loading the saved model
model = Transformer(num_classes, embed_dim, num_heads, num_layers).to(device)
model.load_state_dict(torch.load("best_transformer.pth", map_location=device))

# Setting the model to evaluation mode
model.eval()

# Defining class labels
class_labels = ['assemble_leg', 'drop_drill', 'drop_screw_driver', 'grab_drill', 'take_leg', 'take_screw_driver', 'use_drill', 'use_screw_driver']

# Creating a new directory to save the output images
output_dir = "Output_Class_Predictions"
os.makedirs(output_dir, exist_ok=True)

true_labels_list = []  # True labels
predicted_labels_list = []  # Predicted labels


for i, image in enumerate(test_dataloader):
    image = image.to(device)

    image_name = os.path.splitext(test_dataset.image_files[i])[0]

    # Forward pass of the model
    output = model(image)

    # Getting the predicted class
    _, predicted_class = torch.max(output, 1)

    # Converting tensor label to class label
    predicted_class = class_labels[predicted_class.item()]

    true_label = os.path.basename(os.path.dirname(test_dataset.image_files[i]))  # Extract the true label from subfolder name
    true_labels_list.append(true_label)
    predicted_labels_list.append(predicted_class)

    new_image_name = f"{image_name}_{predicted_class}.jpg"

    image_path = os.path.join(test_data_dir, test_dataset.image_files[i])
    new_image_path = os.path.join(output_dir, new_image_name)
    shutil.copyfile(image_path, new_image_path)

    print(f"Image: {image_name}, True class: {true_label}, Predicted class: {predicted_class}")
    print(f"Renamed image: {new_image_name}")

print("True Labels List:", true_labels_list)
print("Predicted Labels List:", predicted_labels_list)

# Calculating confusion matrix
conf_matrix = confusion_matrix(true_labels_list, predicted_labels_list, labels=class_labels)
print("Confusion Matrix:")
print(conf_matrix)
