import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, kl_weight=1.0):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
    
    def forward(self, recon_x, x, mu, logvar):
        # 재구성 손실 계산
        criterion = nn.MSELoss(reduction='sum')
        MSE = criterion(recon_x, x)
        MSE /= x.size(0)  # 배치 크기로 정규화
        
        # KL 발산 손실 계산
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= x.size(0)  # 배치 크기로 정규화
        
        return MSE + self.kl_weight * KLD

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv_mu = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_logvar = nn.Conv2d(64, 64, kernel_size=1)

        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 이미지 픽셀 값 범위를 [0, 1]로 제한하기 위해 시그모이드 함수 사용
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 표준 정규 분포로부터 샘플링된 랜덤 노이즈
        return mu + std * eps
    
    def forward(self, x):
        x = self.encoder(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar
