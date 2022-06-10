import torch
from torch import nn
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class ChannelWiseFC(nn.Module):
    def __init__(self,  channels, in_features, out_features):
        super().__init__()
        self.channels, self.in_features, self.out_features = channels, in_features, out_features
        
        weights = torch.Tensor(channels, in_features, out_features)
        self.weights = nn.Parameter(weights)
        
        bias = torch.Tensor(channels, out_features)
        self.bias = nn.Parameter(bias)

        nn.init.uniform_(self.weights, -1, 1)
        nn.init.uniform_(self.bias, -1, 1)

    def forward(self, x):
        out = torch.zeros_like(x)

        for i in range(x.shape[0]):
            for j in range(self.channels):
                out[i][j] = torch.add(torch.matmul(x[i][j], self.weights[j].T), self.bias[j])

        return out

class ReconstructiveAutoEncoderChannelWise(nn.Module):
    def __init__(self):
        super(ReconstructiveAutoEncoderChannelWise, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 8, (11, 11), stride= 2, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(8, 32, (5, 5), stride= 2, padding= 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(32, 64, (3, 3), stride= 1, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2),

            nn.Conv2d(64, 128, (3, 3), stride= 1, padding= 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten(-2, -1),

            # Channel-wise fully connected
            ChannelWiseFC(128, 36, 36),
            nn.ReLU(),

            # Decoder
            nn.Unflatten(2, (6, 6)),

            nn.ConvTranspose2d(128, 64, (3, 3), stride= 2, padding= 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, (3, 3), stride= 2, padding= 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 8, (4, 4), stride= 2, padding= 0),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 3, (5, 5), stride= 2, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

summary(ReconstructiveAutoEncoderChannelWise().cuda(), (1, 3, 227, 227))