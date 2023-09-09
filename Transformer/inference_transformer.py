# Inference code for the Transformer for activity recognition
# Provide reference to this code (hamas.khan@tum.de) if using it for your project
# Part of the Sandbox_Hamas repository

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image

class Transformer(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads, num_layers, patch_size):
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

class Inference_Transformer:
    #def __init__(self, model_path, num_classes, embed_dim, num_heads, num_layers, class_labels):
    def __init__(self, model_path, num_classes, embed_dim, num_heads, num_layers, class_labels, patch_size):
        self.model_path = model_path
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.class_labels = class_labels
        self.patch_size = patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        model = Transformer(self.num_classes, self.embed_dim, self.num_heads, self.num_layers, self.patch_size).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model

    @staticmethod
    def _transform_image(image_path, patch_size):
        transform = Compose([
            Resize((patch_size, patch_size)),
            ToTensor(),
        ])
        image = Image.open(image_path).convert("RGB")
        transformed_image = transform(image)
        return transformed_image

    def predict(self, image_path):
        image = self._transform_image(image_path, patch_size=32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted_class = torch.max(output, 1)
            predicted_class = self.class_labels[predicted_class.item()]
        
        return predicted_class

if __name__ == "__main__":
    # Change these configurations according to loaded model. Config can be found in /Hyperparameter Tuning/...
    num_classes = 8
    patch_size = 32
    embed_dim = 256
    num_heads = 2
    num_layers = 1

    model_path = "best_transformer.pth"  # Replace with the model path. Models in /Hyperparameter Tuning/...
    class_labels = ['assemble_leg', 'drop_drill', 'drop_screw_driver', 'grab_drill', 'take_leg', 'take_screw_driver', 'use_drill', 'use_screw_driver']
    image_path = "Test Data/41.jpg"  # Replace with the image path

    #inference_transformer = Inference_Transformer(model_path, num_classes, embed_dim, num_heads, num_layers, class_labels)
    inference_transformer = Inference_Transformer(model_path, num_classes, embed_dim, num_heads, num_layers, class_labels, patch_size)

    predicted_class = inference_transformer.predict(image_path)
    print(f"Predicted Action: {predicted_class}")
