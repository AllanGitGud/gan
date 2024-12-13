import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Hyperparameters
# Learning rates for Generator and Discriminator
learning_rate_G = 5e-5
learning_rate_D = 3e-5  # Lower learning rate for Discriminator
batch_size = 64
# learning_rate = 5e-5
epochs = 50
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data Preparation (Change dataset to FashionMNIST)
transform = transforms.Compose([
    transforms.RandomRotation(15),  # Random rotation between -15 to +15 degrees
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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

# # Generator Model
# class Generator(nn.Module):
#     def __init__(self, latent_dim):     
#         super().__init__()
#         self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
#         self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)  # [n,64, 16, 16]
#         self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)  # [n, 16, 34, 34]
#         self.conv = nn.Conv2d(16, 1, kernel_size=7)  # Output: [n, 1, 28, 28]

#     def forward(self, x):
#         # pass latent space input into the linear layer and reshape
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = x.view(-1, 64, 7, 7)  # 64 feature maps of size 7x7
#         # upsample (transposed conv) 16x16 (64 feature maps)
#         x = self.ct1(x)
#         x = F.relu(x)
#         # upsample to 28x28 (16 feature maps)
#         x = self.ct2(x)
#         x = F.relu(x)
#         # conv to 28x28 (1 feature map)
#         return self.conv(x)

# class Generator(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
#         self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # Nearest-neighbor upsampling
#         self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # Regular convolution
#         self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample again
#         self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # Final output layer
        
#     def forward(self, x):
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = x.view(-1, 64, 7, 7)  # Reshape to 7x7x64
#         x = self.up1(x)
#         x = F.relu(self.conv1(x))
#         x = self.up2(x)
#         return torch.sigmoid(self.conv2(x))  # Final output

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 7*7*128)  # Output a large feature map

        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)  # Final layer to 28x28 image

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)

        x = x.view(-1, 128, 7, 7)  # Reshape to 128 channels, 7x7
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return torch.sigmoid(self.conv3(x))  # Output 28x28 image


# Initialize Models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D)
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G)
criterion = nn.BCELoss()

# Function to compare real and fake images side by side
def compare_images(real, fake, epoch, batch_idx, num_samples=8, save_dir="samples/mnist_fashion"):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Select a subset of real and fake images for display
    real_images = real[:num_samples].cpu().detach()  # Detach the tensor
    fake_images = fake[:num_samples].cpu().detach()  # Detach the tensor

    # Plot real vs fake side by side
    fig, axes = plt.subplots(2, num_samples, figsize=(10, 4))
    for i in range(num_samples):
        axes[0, i].imshow(real_images[i].squeeze(0), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(fake_images[i].squeeze(0), cmap='gray')
        axes[1, i].axis('off')

    # Save the comparison plot
    plt.savefig(f"{save_dir}/e{epoch+1}_b{batch_idx}.png")
    plt.close()

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
        fake_output = discriminator(fake.detach())  # detach to avoid updating generator during this step
        fake_loss = criterion(fake_output, fake_labels)

        # Combine losses and backpropagate
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        # Train Generator after every 2 Discriminator updates
        if batch_idx % 2 == 0:
            optimizer_G.zero_grad()
            fake_output = discriminator(fake)
            loss_G = criterion(fake_output, real_labels)  # Generator wants discriminator to think fake is real
            loss_G.backward()
            optimizer_G.step()

        # Print progress
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(data_loader)}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
            compare_images(real, fake, epoch, batch_idx)

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
