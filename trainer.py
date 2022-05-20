import model
import torch
from torch import nn

net = model.ReconstructiveAutoEncoder()

# 1000 samples, 3 channels, 256x256 pixels
input = torch.rand((1000, 3, 256, 256), requires_grad=True)
target = torch.rand((1000, 3, 256, 256))

before = net.layers[-2].weight.data.clone()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
optimizer.zero_grad()

output = net.forward(input)
lossFunction = nn.MSELoss()
loss = lossFunction(output, target)

loss.backward()
optimizer.step()

print(torch.allclose(net.layers[-2].weight.data, before))