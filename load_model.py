import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from PIL import Image

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)  # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)  # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  # Reshape for ConvTranspose2D
        x = self.ct1(x)
        x = F.relu(x)
        x = self.ct2(x)
        x = F.relu(x)
        return self.conv(x)

# Hyperparameters
latent_dim = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load the pre-trained generator model
generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Load a paint sample as latent noise (optional step)
def load_paint_sample_as_noise(image_path, latent_dim):
    """
    Load an image and convert it to a latent vector.
    Args:
        image_path (str): Path to the paint sample image.
        latent_dim (int): The size of the latent vector.
    Returns:
        torch.Tensor: A latent vector of size [1, latent_dim].
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = TF.resize(image, (latent_dim, 1))  # Resize to match latent dimensions
    image_tensor = TF.to_tensor(image).flatten()  # Flatten to a single vector
    image_tensor = image_tensor[:latent_dim]  # Ensure correct size
    return image_tensor.unsqueeze(0).to(device)

# Example of loading a latent vector from an image
image_path = "./paint_sample.png"
latent_vector = load_paint_sample_as_noise(image_path, latent_dim)

# Generate images
def generate_images(generator, latent_vector, save_dir="generated_samples"):
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        generated_image = generator(latent_vector)
        generated_image = generated_image.cpu().squeeze(0).clamp(0, 1)  # Remove batch dimension and clamp values

        # Save the generated image
        plt.imsave(f"{save_dir}/sample_from_paint.png", generated_image.squeeze(0).numpy(), cmap="gray")

    print(f"Generated image saved in '{save_dir}/sample_from_paint.png'.")

# Generate and save new images using the paint sample
generate_images(generator, latent_vector)
