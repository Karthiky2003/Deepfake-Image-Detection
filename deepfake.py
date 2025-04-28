# ----------------- IMPORTS -----------------
import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm  # For nice progress bars


# ----------------- 1. Dataset Class -----------------
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith(('.jpg', '.png'))]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith(('.jpg', '.png'))]
        self.transform = transform
        self.all_images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ----------------- 2. Transformations -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ----------------- 3. Model -----------------
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # CNN part
        self.cnn = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        
        # ViT part
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        
        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        combined = torch.cat((cnn_features, vit_features), dim=1)
        output = self.fc(combined)
        return output


# ----------------- 4. Train Function -----------------
def train_model(real_dir, fake_dir, num_epochs=10, batch_size=8, learning_rate=1e-4, save_path="deepfake_detector.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Loader
    dataset = DeepfakeDataset(real_dir, fake_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = DeepfakeDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1))

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


# ----------------- 5. Inference -----------------
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    return image

def detect_deepfake(image_path, model_path="deepfake_detector.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        prediction = model(image).item()

    return "Deepfake Detected" if prediction > 0.5 else "Real Image"


# ----------------- 6. Run Training & Inference -----------------
if __name__ == "__main__":
    # Folders
    real_images_folder = r"C:\Users\HP\Downloads\sample_real_fake\real"
    fake_images_folder = r"C:\Users\HP\Downloads\sample_real_fake\fake"
    
    # Train
    train_model(real_images_folder, fake_images_folder, num_epochs=10)

    # Test on a fake sample image
    test_image_path = r"C:\Users\HP\Downloads\sample_real_fake\fake\fake_00.jpg"
    result = detect_deepfake(test_image_path)
    print(f"Result on test image: {result}")
