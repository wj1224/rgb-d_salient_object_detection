from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
import glob
import time
import datetime
import sys
import random

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def get_name(path):
	name, _ = os.path.splitext(os.path.basename(path))
	return name

def load_images(args):
	if args.input_dir is None or not os.path.exists(args.input_dir):
		raise Exception("input path does not exist")

	if args.mode == "train":
		shuffle = True
		transform_list = None

	else:
		shuffle = False
		transform_list = [ 
							transforms.ToTensor(),
						]

	return data.DataLoader(ImageData(args.input_dir, transform_list=transform_list), batch_size=args.batch_size, shuffle=shuffle, num_workers=args.threds)

def tensor2image(tensor):
	image = 255 * (tensor[0].cpu().float().detach().numpy())
	if image.shape[0] == 1:
		image = np.tile(image, (3,1,1))
	return image.astype(np.uint8)

class ImageData(data.Dataset):
	def __init__(self, image_dir, transform_list=None):
		super(ImageData, self).__init__()
		if transform_list is not None:
			self.transform = transforms.Compose(transform_list)
		else:
			self.transform = 0
		
		self.image_path = glob.glob(os.path.join(image_dir, "*.png"))
		if len(self.image_path) == 0:
			image_path = glob.glob(os.path.join(image_dir, "*.jpg"))
		if len(self.image_path) == 0:
			raise Exception("input path contains no images")
		if all(get_name(path).isdigit() for path in self.image_path):
			self.image_path = sorted(self.image_path, key=lambda path: int(get_name))
		else:
			self.image_path = sorted(self.image_path)

	def __getitem__(self, index):
		img = Image.open(self.image_path[index % len(self.image_path)])
		(width, height) = img.size
		RGB_area = (0, 0, width//3, height)
		Depth_area = (width//3, 0, 2*width//3, height)
		GT_area = (2*width//3, 0, width, height)

		RGB_crop = img.crop(RGB_area)
		Depth_crop = img.crop(Depth_area)
		GT_crop = img.crop(GT_area)

		if self.transform == 0:
			SCALE_SIZE = 286
			CROP_SIZE = 256
			
			resize = transforms.Resize(SCALE_SIZE, Image.BICUBIC)
			RGB_images = resize(RGB_crop)
			Depth_images = resize(Depth_crop)
			GT_images = resize(GT_crop)

			i, j, h, w = transforms.RandomCrop.get_params(RGB_images, output_size=(CROP_SIZE, CROP_SIZE))
			RGB_images = transforms.functional.crop(RGB_images, i, j ,h, w)
			Depth_images = transforms.functional.crop(Depth_images, i, j ,h, w)
			GT_images = transforms.functional.crop(GT_images, i, j ,h, w)

			if random.random() > 0.5:
				RGB_images = transforms.functional.hflip(RGB_images)
				Depth_images = transforms.functional.hflip(Depth_images)
				GT_images = transforms.functional.hflip(GT_images)

			RGB_images = transforms.functional.to_tensor(RGB_images)
			Depth_images = transforms.functional.to_tensor(Depth_images)
			GT_images = transforms.functional.to_tensor(GT_images)

		else:
			RGB_images = self.transform(img.crop(RGB_area))
			Depth_images = self.transform(img.crop(Depth_area))
			GT_images = self.transform(img.crop(GT_area))

		return {'RGB': RGB_images, 'Depth': Depth_images, 'GT': GT_images}

	def __len__(self):
		return len(self.image_path)

class Logger():
	def __init__(self, epoch, max_epochs, batches_epochs, tensorboard_dir):
		self.max_epochs = max_epochs
		self.batches_epochs = batches_epochs
		self.epoch = epoch
		self.batch = 1
		self.prev_time = time.time()
		self.mean_period = 0
		self.losses = {}
		self.writer = SummaryWriter(log_dir=tensorboard_dir)

	def log(self, losses=None, images=None):
		self.mean_period += (time.time() - self.prev_time)
		self.prev_time = time.time()
				
		print('Epoch {:04d}/{:04d} [{:04d}/{:04d}]'.format(self.epoch, self.max_epochs, self.batch, self.batches_epochs))

		for i, loss_name in enumerate(losses.keys()):
			if loss_name not in self.losses:
				self.losses[loss_name] = losses[loss_name].item()
			else:
				self.losses[loss_name] += losses[loss_name].item()

			loss_path = '{}'.format(loss_name)
			self.writer.add_scalar(loss_path, self.losses[loss_name] / self.batch, (self.epoch - 1) * self.batches_epochs + self.batch)
			print('{:s}: {:.4f}'.format(loss_name, self.losses[loss_name] / self.batch))
			
		for image_name, tensor in images.items():
			image_path = '{}'.format(image_name)
			image_show = tensor2image(tensor)
			self.writer.add_image(image_path, image_show, (self.epoch - 1) * self.batches_epochs + self.batch)

		if (self.batch % self.batches_epochs) == 0:
			for loss_name, loss in self.losses.items():
				self.losses[loss_name] = 0.0
			self.epoch += 1
			self.batch = 1
			print('\n')
			
		else:
			self.batch += 1
			print('\n')
