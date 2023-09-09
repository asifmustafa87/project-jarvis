'''
Training a Transformer model from scratch on a large-scale dataset can be computationally intensive and time-consuming.
However, here we are using a pre-trained ResNet backbone and applying the Transformer layers on top for transfer learning. 
This approach allows us to benefit from the pre-trained ResNet weights for feature extraction and 
training the layers for our specific dataset.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import torchvision.models as models

class Transformer(nn.Module):
    def __init__(self, num_classes, num_frames, embed_dim, num_heads, num_layers):
        super(Transformer, self).__init__()

        self.num_frames = num_frames

        # Pre-trained ResNet backbone
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

        # Transformer Layers
        self.temporal_embedding = nn.Parameter(torch.randn(num_frames, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, num_frames, _, _, _ = x.shape

        x = x.view(batch_size * num_frames, 3, 224, 224)
        x = self.backbone(x)
        x = x.view(batch_size, num_frames, -1)

        x = x + self.temporal_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)

        x = self.fc(x)

        return x


class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, _ = self.data[idx]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return torch.stack(frames)


# Hyperparameters
num_classes = 9
num_frames = 16
embed_dim = 256
num_heads = 4
num_layers = 2
batch_size = 8
learning_rate = 1e-4
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the transformation pipeline
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

data_dir = "path/to/dataset"
dataset = VideoDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Creating the model and moving it to the device
model = Transformer(num_classes, num_frames, embed_dim, num_heads, num_layers).to(device)

# Defining the loss function and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Determining the split size (you can adjust this ratio)
val_split = 0.2
val_size = int(val_split * len(dataset))
train_size = len(dataset) - val_size

# Splitting the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Creating data loaders for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

total_steps = len(dataloader)
for epoch in range(num_epochs):
    for i, frames in enumerate(dataloader):
        frames = frames.to(device)

        # Forward pass instance
        outputs = model(frames)

        # Computing the loss
        labels = torch.zeros(frames.size(0)).long().to(device)  # Placeholder labels
        loss = criterion(outputs, labels)

        # Backward propagtion pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "saved_model.pth")
