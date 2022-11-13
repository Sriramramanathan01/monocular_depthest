import cv2
import torch
import torchvision.transforms as transforms
import time
import numpy as np
from PIL import Image
import torch.nn as nn

def get_midas(device):
    # Load a MiDas model for depth estimation
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move model to GPU if available
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        midas_transform = midas_transforms.dpt_transform
    else:
        midas_transform = midas_transforms.small_transform

    return midas, midas_transform


def midas_pass(arr, midas, midas_transform, device, batch_size):
    out = None
    for i in range(batch_size):
        array = arr[i].permute(1, 2, 0)
        array = array.numpy()
        input_batch = midas_transform(array).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=array.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()

        im = Image.fromarray(output)
        if im.mode != 'RGB':
            im = im.convert('RGB')

        trans_im_to_tensor = transforms.ToTensor()
        o_p = trans_im_to_tensor(im)
        o_p = torch.unsqueeze(o_p, 0)

        if out == None:
            out = o_p
        else:
            out = torch.cat((out, o_p), 0)
    return out
