import model
import torch
from torch import nn
from DataManager import LoadFromFolder
from torchvision import transforms
from visualizer import visualizeReconstruction, visualizeTensor

#################################################### LOAD DATA

transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.ToTensor()
                        ])

targetSetPath = "./data/paris_complete"
inputSetPath = "./data/paris_corrupted"

targetDataset = LoadFromFolder(targetSetPath, transform=transform)
targetDataloader = torch.utils.data.DataLoader(targetDataset)

inputDataset = LoadFromFolder(inputSetPath, transform=transform)
inputDataloader = torch.utils.data.DataLoader(inputDataset)

input = next(iter(inputDataloader))
target = next(iter(targetDataloader))

################################################################

net = model.ReconstructiveAutoEncoder()

temp = 0

for i, (input, target) in enumerate(zip(iter(inputDataloader), iter(targetDataloader))):
    before = net.layers[-2].weight.data.clone()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    optimizer.zero_grad()

    output = net.forward(input)
    lossFunction = nn.MSELoss()
    loss = lossFunction(output, target)

    loss.backward()
    optimizer.step()

    print(torch.allclose(net.layers[-2].weight.data, before))

    if i == 99:
        visualizeTensor(output[0])
        print(output[0])