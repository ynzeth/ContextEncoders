import model_channelWise
import torch
from torch import nn
from DataManager import LoadFromFolder
from torchvision import transforms
from visualizer import pasteInpainting, visualizeTensor
from torchvision.transforms import functional

BATCHSIZE = 32

############################################# LOAD DATA AND NETWORK

input_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.ToTensor()
                        ])

target_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.CenterCrop(99),
                        transforms.ToTensor()
                        ])

path = "./data"

inputDataset = LoadFromFolder(path, transform=input_transform)
inputDataloader = torch.utils.data.DataLoader(inputDataset, batch_size = BATCHSIZE)

targetDataset = LoadFromFolder(path, transform=target_transform)
targetDataloader = torch.utils.data.DataLoader(targetDataset, batch_size = BATCHSIZE)

net = model_channelWise.ReconstructiveAutoEncoderChannelWise()

#################################################################

################################################### TRAINING LOOP

optimizer = torch.optim.Adam(net.parameters(), 0.001)

for i, (input, target) in enumerate(zip(inputDataloader, targetDataloader)):
    before = net.layers[-13].weights[0].data.clone()

    optimizer.zero_grad()

    input = functional.erase(input, 65, 65, 99, 99, 0)
    output = net.forward(input)

    lossFunction = nn.L1Loss()
    loss = lossFunction(output, target)

    loss.backward()
    optimizer.step()

    print(torch.allclose(before, net.layers[-13].weights[0].data))

    print(i, torch.sum(loss).item())

################################################################

########################################### VALIDATION

validationPath = "./validation"

validationDataset = LoadFromFolder(validationPath, transform=input_transform)
validationloader = torch.utils.data.DataLoader(validationDataset, batch_size = 10)

input = next(iter(validationloader))
input = functional.erase(input, 65, 65, 99, 99, 0)
output = net.forward(input)

show = pasteInpainting(output[0], input[0])
for i, o in enumerate(output):
    if i > 0:
        show = torch.cat((show, pasteInpainting(o, input[i])), 2)

visualizeTensor(show)