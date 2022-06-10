import torch
from torch import nn
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class ReconstructiveAutoEncoder(nn.Module):
    def __init__(self):
        super(ReconstructiveAutoEncoder, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 4, (11, 11), stride= 2, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(4, 8, (5, 5), stride= 1, padding= 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(8, 32, (3, 3), stride= 1, padding= 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),
            
            nn.Flatten(-3, -1),

            # Fully connected
            nn.Linear(4608,  4608),
            nn.ReLU(),

            # Decoder
            nn.Unflatten(1, (32, 12, 12)),
            nn.ConvTranspose2d(32, 8, (3, 3), stride= 2, padding= 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, (4, 4), stride= 2, padding= 0),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, (5, 5), stride= 2, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

summary(ReconstructiveAutoEncoder().cuda(), (1, 3, 227, 227))