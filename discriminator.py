import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return torch.sigmoid(self.fc2(x))

# Initialize the model
model = Discriminator()

# Create dummy data (batch of 16 MNIST-like images)
batch_size = 2
dummy_data = torch.randn(batch_size, 1, 28, 28)  # [batch_size, channels, height, width]

# Visualize the first image in the batch
image = dummy_data[0, 0].detach().numpy()  # Select first image and convert to numpy for plotting

plt.imshow(image, cmap="gray")
plt.title("Dummy MNIST Image")
plt.colorbar()
plt.show()

# Pass the dummy data through the model
output = model(dummy_data)

# Print input and output shapes
print(f"Input shape: {dummy_data.shape}")
print(f"Output shape: {output.shape}")
print(f"Output values: {output}")



