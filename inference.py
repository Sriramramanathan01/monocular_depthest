import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from model import get_student
import cv2
import PIL

model = get_student()
model.load_state_dict((torch.load("train_data/inference_network.pth")))
model.eval()

transform1 = transforms.ToTensor()
transform2 = transforms.ToPILImage()

inp = cv2.imread('inputs/53.png')
inp_tensor = transform1(inp)
inp_tensor = torch.unsqueeze(inp_tensor, dim=0)

out = model(inp_tensor)
output = torch.squeeze(out[0], dim=0)
out_img = transform2(output)
out_img.save("outputs/out.png")
