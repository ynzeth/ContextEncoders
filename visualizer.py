from torchvision import transforms
from matplotlib import pyplot as plt

ROWS, COLUMNS = 1, 3

def visualizeReconstruction(inputTensor, outputTensor, targetTensor):
    tensor_to_img = transforms.ToPILImage()
    inputImg = tensor_to_img(inputTensor)
    outputImg = tensor_to_img(outputTensor)
    targetImg = tensor_to_img(targetTensor)
    
    fig = plt.figure(figsize=(300, 900))

    fig.add_subplot(ROWS, COLUMNS, 1)
    inputImg.show()

    fig.add_subplot(ROWS, COLUMNS, 2)
    outputImg.show()

    fig.add_subplot(ROWS, COLUMNS, 3)
    targetImg.show()

def visualizeTensor(imageTensor):
    tensor_to_img = transforms.ToPILImage()
    
    img = tensor_to_img(imageTensor)
    img.show()
