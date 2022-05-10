import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class ReconstructiveAutoEncoder(nn.Module):
    def __init__(self):
        super(ReconstructiveAutoEncoder, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 3, (4, 4), stride= 2, padding= 1),
            nn.ReLU(),
            nn.Conv2d(3, 2, (4, 4), stride= 2, padding= 1),
            nn.ReLU(),
            nn.Conv2d(2, 1, (4, 4), stride= 2, padding= 1),
            nn.Flatten(),

            # Fully connected
            nn.Linear(1024,  1024),
            nn.ReLU(),

            # Decoder
            nn.Unflatten(1, (32, 32)),
            nn.ConvTranspose2d(1, 2, (4, 4), stride= 2, padding= 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 3, (4, 4), stride= 2, padding= 1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, (4, 4), stride= 2, padding= 1),
            nn.ReLU()
        )

    def forward(self, x):
        for layer in self.layers:
            print(layer, "| input shape :" , x.shape)
            x = layer.forward(x)

        return x