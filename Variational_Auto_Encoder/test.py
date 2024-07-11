import VAE
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 로드 및 GPU로 이동
model = VAE.VAE()
model.load_state_dict(torch.load('autoencoder.pth'))
model.to(device)
model.eval()

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

validation_data_path= "/home/asl/projects/AutoEncoder/dataset/validation_processed/"
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

validation_dataset = CustomDataset(data_dir=validation_data_path, transform=data_transforms)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# 이미지 재구성 결과 시각화
with torch.no_grad():
    for images in validation_loader:
        images = images.to(device)
        outputs, mus, logvars = model(images)
        
        # 텐서를 numpy 배열로 변환
        images_np = images.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        
        # 원본 및 재구성된 이미지 시각화
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, img, title in zip(axes, [images_np, outputs_np], ['Original', 'Reconstructed']):
            ax.imshow(np.transpose(img, (0, 2, 3, 1))[0])
            ax.axis('off')
            ax.set_title(title)
        plt.tight_layout()
        plt.pause(2)  # 2초 동안 창을 열어둠
        plt.close()
