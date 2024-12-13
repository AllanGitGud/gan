import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Hyperparameters
batch_size = 64
learning_rate = 3e-4
epochs = 50
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data Preparation (Change dataset to FashionMNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize FashionMNIST images
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(320, 50)  # Adjust to match flattened dimensions
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return torch.sigmoid(self.fc2(x))

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim):     
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)  # [n,64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)  # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # Output: [n, 1, 28, 28]

    def forward(self, x):
        # pass latent space input into the linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  # 64 feature maps of size 7x7
        # upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)
        # upsample to 28x28 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)
        # conv to 28x28 (1 feature map)
        return self.conv(x)

# Initialize Models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# Training Loop
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(data_loader):
        # Prepare real and fake data
        real = real.to(device)  # Shape: [batch_size, 1, 28, 28]
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
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

print("Models saved successfully!")

# Generate and Save Samples
def save_samples(generator, num_samples=16, latent_dim=100, save_dir="samples/fashion"):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        fake_images = generator(noise)
        fake_images = fake_images.cpu().squeeze(1).clamp(0, 1)  # Remove channel dimension and clamp values
        for i, img in enumerate(fake_images):
            plt.imsave(f"{save_dir}/sample_{i+1}.png", img.numpy(), cmap="gray")

save_samples(generator)
