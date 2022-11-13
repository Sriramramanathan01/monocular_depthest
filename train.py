import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataset import return_loader
from model import get_student
from midas_test import get_midas, midas_pass
import datetime
from tqdm import tqdm

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = return_loader()

student = get_student()
midas, midas_transform = get_midas(device)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

#Hyper-parameters
lr = 3e-4
num_epochs = 100
distill_loss = nn.MSELoss()
training_loss = RMSELoss()
optimizer = optim.Adam(student.parameters(), lr=lr, betas=[0.9, 0.999], eps=1e-3)
alpha = 0.1
def training_loop(n_epochs, optimizer, distill_loss, training_loss, train_loader):
  losses = []
  for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (rgb, depth) in loop:
      #teacher = dpt_model.to(device)
      soft_target = midas_pass(rgb, midas, midas_transform, device, 1)
      soft_target = soft_target.to(device)
      rgb = rgb.to(device)
      depth = depth.to(device)
      student.to(device)
      intermediete_out, output = student(rgb)

      dist_loss = distill_loss(intermediete_out, soft_target)
      student_loss = training_loss(output, depth)

      total_loss = (1 - alpha)*student_loss + (alpha * dist_loss)

      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

      loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
      loop.set_postfix(loss = total_loss.item())
      train_loss += total_loss.item()
    if epoch > 1:
      losses.append(train_loss / len(train_loader))
    torch.save(student.state_dict(), '/home/pytorch/anaconda3/envs/sriram/depth_est/train_data/inference_network.pth')
  plt.title("Training loss")
  plt.plot(losses,label="train")
  plt.xlabel("Iterations") #Change to 'epochs' next time
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig('/home/pytorch/anaconda3/envs/sriram/depth_est/train_data/loss.png')
  #plt.show()

training_loop(
    n_epochs=num_epochs,
    optimizer = optimizer,
    distill_loss = distill_loss,
    training_loss = training_loss,
    train_loader = train_loader
)
