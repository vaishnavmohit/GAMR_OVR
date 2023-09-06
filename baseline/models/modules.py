import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

# from util import log

import logging
log = logging.getLogger(__name__)

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels, stride=1)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Down_s(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.stride_conv = nn.Sequential(
			# nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels, stride=2)
		)

	def forward(self, x):
		return self.stride_conv(x)


class Enc_Conv_v0_16(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 64x224x224
		self.down1 = Down(64, 64) # 64x128x128
		self.down1s = DoubleConv(64, 64) # 64x112x112
		self.down2 = Down(64, 128) # 128x64x64
		self.down2s = DoubleConv(128, 128) # 128x56x56
		self.down3 = Down(128, 128) # 128x32x32
		self.down3s = DoubleConv(128, 128) # 128x28x28
		self.down4 = Down(128, 128) # 128x14x14 
		self.down4s = DoubleConv(128, 256) # 128x14x14
		self.down5 = Down(256, 256) # 128x7x7
		self.down5s = DoubleConv(256, 256) # 128x7x7 
		# self.down6 = Down(256, 256) # 128x3x3
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		
		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], 128, -1), 2, 1) # B, C, HW

		return z_kv 


class Resnet_block(nn.Module):
	def __init__(self, pretrained = False):
		super().__init__()
		
		self.value_size = 128 #args.value_size
		self.key_size = 128 #args.key_size
		# Inputs to hidden layer linear transformation
		self.resnet = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
		self.conv_kv = nn.Conv2d(2048, self.value_size+self.key_size, 1, stride=1, padding=0)
		self.relu = nn.ReLU()

		
	def forward(self, x):
		# Pass the input tensor through each of our operations
		x = self.resnet(x) # input = 1024
		z_kv = self.relu(self.conv_kv(x))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
		z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)

		return z_keys, z_values 