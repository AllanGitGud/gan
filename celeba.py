import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
learning_rate = 3e-4
epochs = 50
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data Preparation for CelebA dataset
transform = transforms.Compose([
    transforms.Resize(64),  # Resize to 64x64
    transforms.CenterCrop(64),  # Crop to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images with single channel
])

# dataset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Discriminator Model
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # 64x64x3 -> 32x32x64
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 32x32x64 -> 16x16x128
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 16x16x128 -> 8x8x256
#         self.fc1 = nn.Linear(8*8*256, 1)  # Flatten and output single value for real/fake

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x), 0.2)
#         x = F.leaky_relu(self.conv2(x), 0.2)
#         x = F.leaky_relu(self.conv3(x), 0.2)
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = torch.sigmoid(self.fc1(x))  # Sigmoid output for binary classification (real/fake)
#         return x

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim):     
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 64 * 64)  # Output size: 64x64x3 (RGB image)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 3, 64, 64)  # Reshape to 64x64x3
        return torch.tanh(x)  # Output image should be in range [-1, 1]

# Initialize Models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training Loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(data_loader):
        # Prepare real and fake data
        real = real.to(device)  # Shape: [batch_size, 3, 64, 64]
        batch_size = real.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Generate fake data (random noise)
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake = generator(noise)

        # Train Discriminator on real data
        optimizer_D.zero_grad()
        real_output = discriminator(real)
        real_loss = criterion(real_output, real_labels)

        # Train Discriminator on fake data
        fake_output = discriminator(fake.detach())
        fake_loss = criterion(fake_output, fake_labels)

        # Combine losses and backpropagate
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_output = discriminator(fake)
        loss_G = criterion(fake_output, real_labels)  # Generator wants discriminator to think fake is real
        loss_G.backward()
        optimizer_G.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(data_loader)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

print("Training complete!")

# Save the trained generator and discriminator
torch.save(generator.state_dict(), "generator_celeba.pth")
torch.save(discriminator.state_dict(), "discriminator_celeba.pth")

print("Models saved successfully!")

# Generate and Save Samples
def save_samples(generator, num_samples=16, latent_dim=100, save_dir="samples/fashion_mnist"):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        fake_images = generator(noise)
        fake_images = fake_images.cpu().clamp(0, 1)  # Clamp values to [0, 1] range
        for i, img in enumerate(fake_images):
            plt.imsave(f"{save_dir}/sample_{i+1}.png", img.permute(1, 2, 0).numpy())  # Save image as RGB

save_samples(generator)
