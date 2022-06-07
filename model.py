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
            nn.Conv2d(3, 16, (11, 11), stride= 4, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(16, 32, (5, 5), stride= 1, padding= 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(32, 64, (3, 3), stride= 1, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),
            
            # nn.Flatten(),

            # Fully connected
            # nn.Linear(1024,  1024),
            # nn.ReLU(),

            # Decoder
            # nn.Unflatten(1, (1, 32, 32)),
            nn.ConvTranspose2d(64, 32, (3, 3), stride= 3, padding= 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride= 3, padding= 0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (5, 5), stride= 2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        for layer in self.layers:
            # print(layer, "| input shape :" , x.shape)
            x = layer.forward(x)

        return x

summary(ReconstructiveAutoEncoder().cuda(), (3, 227, 227))