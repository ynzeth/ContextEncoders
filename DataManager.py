import os
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
from torchvision import datasets, transforms
import torch

### How To Use ###

# 1: Define and compose transforms: 

# transform = transforms.Compose([transforms.Resize(227),
#                         transforms.CenterCrop(226),
#                         transforms.ToTensor(),
#                         transforms.RandomErasing(p=1, scale=(0.25, 0.25), ratio=(1, 1), inplace=False)])

# 2: instantiate  loadFormFolder class:

# dataset = LoadFromFolder(evalSetPath, transform=transform)

# 3 DataLoader wraps an iterable around the Dataset
# eval_dataloader = torch.utils.data.DataLoader(dataset)


class LoadFromFolder(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
         
        # List all images in folder and count them
        all_imgs = os.listdir(main_dir)
        # self.total_imgs = natsorted(all_imgs)
        self.total_imgs = all_imgs
    def __len__(self):
        # Return the previously computed number of images
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        # Apply the transformations
        tensor_image = self.transform(image)
        return tensor_image 

# test Create_DataLoader
# evalSetPath = "C:/Users/korat/Desktop/BME/Semester 2/Computer Vision/paris_data/paris_eval"
# trainSetPath = "C:/Users/korat/Desktop/BME/Semester 2/Computer Vision/paris_data/paris_train"

# dataset = LoadFromFolder(evalSetPath, transform=transform)
# eval_dataloader = torch.utils.data.DataLoader(dataset)
# print(next(iter(eval_dataloader)).shape)  # prints shape of image with single batch

# tensor_to_img = transforms.ToPILImage()
# img = tensor_to_img(next(iter(eval_dataloader))[0,:,:,:])

# img.show()