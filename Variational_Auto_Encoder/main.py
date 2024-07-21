import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import VAE

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

train_data_path= "/Users/jinwoolee/Projects/AutoEncoder/dataset/train_processed/"
validation_data_path= "/Users/jinwoolee/Projects/AutoEncoder/dataset/validation_processed/"

model = VAE.VAE()
criterion = VAE.VAELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(data_dir=train_data_path, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

validation_dataset = CustomDataset(data_dir=validation_data_path, transform=data_transforms)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)

num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델과 손실 함수를 GPU로 이동
model.to(device)
criterion.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images = data.to(device)

        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)  # VAE 모델의 출력 처리
        loss = criterion(recon_images, images, mu, logvar)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:    # Print every 10 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        for images in validation_loader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            val_loss += criterion(recon_images, images, mu, logvar).item()
        
        val_loss /= len(validation_loader)
        print(f'Validation Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'autoencoder.pth')

print("Training complete.")
