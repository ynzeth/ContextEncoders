import model
import torch

net = model.ReconstructiveAutoEncoder()

input = torch.rand((3, 256, 256))
output = net.forward(input)

print(output.shape)