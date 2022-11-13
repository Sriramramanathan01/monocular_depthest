import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


#Lighter Student Network
class conv_bn(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.bn(self.conv1(x)))


class conv_dw(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))


class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.enc_block = nn.ModuleList([
        conv_bn(3, 32, 2), 
        conv_dw(32, 64, 1),
        conv_dw(64, 128, 2),
        conv_dw(128, 256, 2),
        conv_dw(256, 512, 2),
        conv_dw(512, 1024, 2)])

  def forward(self, x):
    x = F.interpolate(x, (128, 160))
    ftrs = []
    for block in self.enc_block:
      x = block(x)
      ftrs.append(x)
    ftrs.reverse()
    return ftrs


class feature_fusion(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.fus_branch = nn.Sequential(
        nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride = 1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1))

  def forward(self, x):
    x = self.fus_branch(x)
    return x


class Decoder(nn.Module):
  def __init__(self, chs=(1024, 512, 256, 128, 64)):
    super().__init__()
    self.chs = chs
    self.dec_block = nn.ModuleList([feature_fusion(chs[i], chs[i+1]) for i in range(len(self.chs)-1)])
    self.fus_transpconv = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=3, stride=1, padding=1) for i in range(len(self.chs)-1)])

    self.transp1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
    self.conv1 = nn.Conv2d(32, 32, 1)
    self.transp2 = nn.ConvTranspose2d(32, 3, 3, padding=1)
    self.conv2 = nn.Conv2d(3, 3, 1)
    self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    self.scale = nn.UpsamplingBilinear2d([480, 640])

  def forward(self, x, enc_features):
    #outs = []
    for i in range(len(self.chs)-1):
      x = self.dec_block[i](x)
      x = torch.cat([x, enc_features[i+1]], dim=1)
      x = self.fus_transpconv[i](x)
    x = self.conv1(self.transp1(x))
    x = torch.cat((x, enc_features[len(self.chs)]), dim=1)
    x = self.conv1(self.transp1(x))
    x = self.conv2(self.transp2(x))
    x = self.up(x)
    x = self.scale(x)
    return x


#Putting it together
class light_net(nn.Module):
  def __init__(self, dec_chs = (1024, 512, 256, 128, 64)):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder(dec_chs)
    self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
    self.conv2 = nn.Conv2d(3, 3, 1)
    self.ReLU = nn.ReLU()

  def forward(self, x):
    enc_features = self.encoder(x)
    dec_out = self.decoder(enc_features[0], enc_features)
    out = self.conv1(dec_out)
    #out = self.ReLU(out)
    out = self.conv2(out)
    out = self.ReLU(out)
    return dec_out, out


#Teacher network: Vision Transformer DPT

def get_student():
  #Student network
  student = light_net()
  return student

