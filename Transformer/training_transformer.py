# Code for transfer learning of transformer for activity recognition
# Provide reference to this code (hamas.khan@tum.de) if using it for your project
# Part of the Sandbox_Hamas repository

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import torchvision

class Transformer(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads, num_layers, patch_size, dropout_prob):
        super(Transformer, self).__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size

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
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        batch_size, _, _, _ = x.shape

        x = self.embedding(x)
        x = x.permute(2, 3, 0, 1)

        seq_length = x.shape[2]
        x = x.reshape(-1, seq_length, self.embed_dim)

        x = self.transformer(x)

        x = x.mean(dim=0)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.labels = [self.data.classes[label] for _, label in self.data.samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, label = self.data[idx]
        return frame, label

def Loss_Plot(losses, val_losses):
    fig, ax = plt.subplots()
    ax.plot(range(len(losses)), losses, 'b-', label='Training Loss')
    ax.plot(range(len(val_losses)), val_losses, 'r-', label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    plt.savefig('training_validation_loss.png')
    plt.close()

# Hyperparameters
num_classes = 8
patch_size = 32
embed_dim = 256
num_heads = 2
num_layers = 1
batch_size = 8
learning_rate = 1e-3
num_epochs = 100
dropout_prob = 0.5
l2_lambda = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation Pipeline
transform = Compose([
    Resize((patch_size, patch_size)),
    ToTensor(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
])

# Loading the pre-trained model weights for Transfer Learning
pretrained_model_path = "pretrained_8x32_224_K400.pyth"
pretrained_model = Transformer(num_classes, embed_dim, num_heads, num_layers, patch_size, dropout_prob).to(device)

# Enable GPU on CUDA supported platform during training. Only CPU enabled on home Laptop.
pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')), strict=False)
# pretrained_model.load_state_dict(torch.load(pretrained_model_path), strict=False)   # Disabling mapping to CPU only

# Replacing the final fully connected layer with a new one for the specific number of classes
pretrained_model.fc = nn.Linear(embed_dim, num_classes).to(device)

# Selecting Adam optimizer for the whole parameters and adding L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = Adam(pretrained_model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

# Loading the Dataset and Splitting into Training and Validation Sets
data_dir = "Training Data"
dataset = VideoDataset(data_dir, transform=transform)

# Determining the split size (you can adjust this ratio)
val_split = 0.2
val_size = int(val_split * len(dataset))
train_size = len(dataset) - val_size

# Splitting the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Creating data loaders for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training Loop with Validation
train_losses = []  # List to store the training losses
val_losses = []    # List to store the validation losses

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(num_epochs):
    # Training Phase
    pretrained_model.train()  # Set the model to training mode
    train_epoch_loss = 0.0

    for i, (frames, labels) in enumerate(train_dataloader):
        frames = frames.to(device)
        labels = labels.to(device)

        # This is the Forward Pass on the model
        outputs = pretrained_model(frames)

        # Computing the loss between model inference and target labels
        loss = criterion(outputs, labels)

        # Regularization - L2 regularization for all trainable parameters
        l2_reg = torch.tensor(0.).to(device)
        for param in pretrained_model.parameters():
            l2_reg += torch.norm(param)

        loss += l2_lambda * l2_reg

        # Backward pass/propagation and optimization function call
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    train_epoch_loss = train_epoch_loss / len(train_dataloader)
    train_losses.append(train_epoch_loss)

    # Validation Phase
    with torch.no_grad():
        pretrained_model.eval()
        val_epoch_loss = 0.0

        for frames, labels in val_dataloader:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = pretrained_model(frames)
            loss = criterion(outputs, labels)

            val_epoch_loss += loss.item()

        val_epoch_loss = val_epoch_loss / len(val_dataloader)
        val_losses.append(val_epoch_loss)

    # Plotting the training and validation losses after each epoch
    Loss_Plot(train_losses, val_losses)

    if (epoch + 1) % 100 == 0:
        # Saving the model after every 100 epochs
        save_path = f"transformer_epoch{epoch+1}.pth"
        torch.save(pretrained_model.state_dict(), save_path)
        print(f"Model Saved at Epoch {epoch+1}")

    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")

# Saving the fine-tuned model after training completion
torch.save(pretrained_model.state_dict(), "transformer_final.pth")
print("Training Completed. Model Saved!")
