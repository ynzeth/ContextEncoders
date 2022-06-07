import model
import torch
from torch import nn
from DataManager import LoadFromFolder
from torchvision import transforms
from visualizer import visualizeReconstruction, visualizeTensor

#################################################### LOAD DATA AND NET

input_transform = transforms.Compose([
                        transforms.ToTensor()
                        ])

target_transform = transforms.Compose([
                        transforms.CenterCrop(99),
                        transforms.ToTensor()
                        ])

inputSetPath = "./data/paris_corrupted"
targetSetPath = "./data/paris_complete"

inputDataset = LoadFromFolder(inputSetPath, transform=input_transform)
inputDataloader = torch.utils.data.DataLoader(inputDataset)

targetDataset = LoadFromFolder(targetSetPath, transform=target_transform)
targetDataloader = torch.utils.data.DataLoader(targetDataset)

net = model.ReconstructiveAutoEncoder()

################################################################

# for i, (input, target) in enumerate(zip(iter(inputDataloader), iter(targetDataloader))):
#     before = net.layers[-2].weight.data.clone()

#     optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
#     optimizer.zero_grad()

#     output = net.forward(input)
#     lossFunction = nn.MSELoss()
#     loss = lossFunction(output, target)

#     loss.backward()
#     optimizer.step()

#     print(torch.allclose(net.layers[-2].weight.data, before))

#     if i == 99:
#         visualizeTensor(output[0])
#         print(output[0])

###################################### OVERFITTING ON FIRST SAMPLE

input = next(iter(inputDataloader))
target = next(iter(targetDataloader))

for i in range (100000):
    before = net.layers[-2].weight.data.clone()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    optimizer.zero_grad()

    output = net.forward(input)
    lossFunction = nn.MSELoss()
    loss = lossFunction(output, target)

    loss.backward()
    optimizer.step()

    print(torch.allclose(net.layers[-2].weight.data, before))

    if i == 99999:
        visualizeTensor(output[0])

print(output[0])
print(target[0])

#####################################################