import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import cv2

path_rgb = '/media/data/sriram/mobiledepth/rgb/*.*'
path_depth = '/media/data/sriram/mobiledepth/depth/*.*'
rgb_list = []
depth_list = []
for file in glob.glob(path_rgb):
  rgb_list.append(file)

for file in glob.glob(path_depth):
  depth_list.append(file)

rgb_list.sort()
depth_list.sort()

#Defining the dataset class
class depth_est_dataset(Dataset):
  def __init__(self, rgb_list, depth_list, transform=None):
    self.rgb_list = rgb_list
    self.depth_list = depth_list
    self.transform = transform

  def __len__(self):
    return len(self.rgb_list)

  def __getitem__(self, index):
    rgb_path = self.rgb_list[index]
    depth_path = self.depth_list[index]
    rgb_img = cv2.imread(rgb_path)
    depth_img = cv2.imread(depth_path)

    if self.transform:
      rgb_img = self.transform(rgb_img)
      depth_img = self.transform(depth_img)

    return rgb_img, depth_img


def return_loader():
    dataset = depth_est_dataset(rgb_list, depth_list, transform = transforms.Compose([transforms.ToTensor()]))
    train_set, test_set = torch.utils.data.random_split(dataset, [6800, 585])
    train_loader = DataLoader(train_set, batch_size=1, shuffle = True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle = True)
    return train_loader, test_loader
