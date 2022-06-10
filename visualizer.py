from torchvision import transforms

def visualizeTensor(imageTensor):
    tensor_to_img = transforms.ToPILImage()
    
    img = tensor_to_img(imageTensor)
    img.show()

def pasteInpainting(inpainting, input):
    for i in range(99):
        for j in range(99):
            input[0][65+i][65+j] = inpainting[0][i][j]
            input[1][65+i][65+j] = inpainting[1][i][j]
            input[2][65+i][65+j] = inpainting[2][i][j]

    return input