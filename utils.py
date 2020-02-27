import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
import glob

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def load_images(args):
	if args.input_dir is None or not os.path.exists(args.input_dir):
		raise Exception("input path does not exist")

	transform_list = []
	transform_list.append(transforms.ToTensor())
#	transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

	return data.DataLoader(DatasetGen(args.input_dir, args.mode, transform_list=transform_list),
							batch_size=args.batch_size,
							shuffle=args.mode=="train",
							num_workers=args.threds)

def tensor2image(tensor):
	image = 255 * (tensor[0].cpu().float().detach().numpy())
	if image.shape[0] == 1:
		image = np.tile(image, (3,1,1))
	return image.astype(np.uint8)

class DatasetGen(data.Dataset):
	def __init__(self, image_dir, mode="train", transform_list=None):
		super(DatasetGen, self).__init__()
		self.transform = transforms.Compose(transform_list)

		self.image_path = []

		self.image_path.append(glob.glob(os.path.join(image_dir, "rgb", "*.png")))
		self.image_path.append(glob.glob(os.path.join(image_dir, "depth", "*.png")))
		if mode == "train":
			self.image_path.append(glob.glob(os.path.join(image_dir, "gt", "*.png")))

		if len(self.image_path) == 0:
			self.image_path.append(glob.glob(os.path.join(image_dir, "rgb", "*.jpg")))
			self.image_path.append(glob.glob(os.path.join(image_dir, "depth", "*.jpg")))
			if mode == "train":
				self.image_path.append(glob.glob(os.path.join(image_dir, "gt", "*.jpg")))

		if len(self.image_path) == 0:
			raise Exception("input path contains no images, only support .png or .jpg images")

		for i in range(len(self.image_path)):
			self.image_path[i].sort()

	def __getitem__(self, index):
		rgb = Image.open(self.image_path[0][index % len(self.image_path[0])])
		rgb = self.transform(rgb)
		depth = Image.open(self.image_path[1][index % len(self.image_path[0])])
		depth = transforms.functional.to_tensor(depth)
		if len(self.image_path) == 3:
			gt = Image.open(self.image_path[2][index % len(self.image_path[0])])
			gt = transforms.functional.to_tensor(gt)

		if len(self.image_path) == 3:
			return {'RGB': rgb, 'Depth': depth, 'GT': gt}
		else:
			return {'RGB': rgb, 'Depth': depth}

	def __len__(self):
		return len(self.image_path[0])

class Logger():
	def __init__(self, epoch, max_epochs, max_batches, log_dir):
		self.max_epochs = max_epochs
		self.max_batches = max_batches
		self.epoch = epoch
		self.batch = 1
		self.losses = {}
		self.writer = SummaryWriter(log_dir=log_dir)

	def log(self, losses=None, images=None):
		print('Epoch {:04d}/{:04d} [{:04d}/{:04d}]'.format(self.epoch, self.max_epochs, self.batch, self.max_batches))

		for i, loss_name in enumerate(losses.keys()):
			if loss_name not in self.losses:
				self.losses[loss_name] = losses[loss_name].item()
			else:
				self.losses[loss_name] += losses[loss_name].item()

			loss_path = '{}'.format(loss_name)
			self.writer.add_scalar(loss_path, self.losses[loss_name] / self.batch, (self.epoch - 1) * self.max_batches + self.batch)
			print('{:s}: {:.4f}'.format(loss_name, self.losses[loss_name] / self.batch))
			
		for image_name, tensor in images.items():
			image_path = '{}'.format(image_name)
			image_show = tensor2image(tensor)
			self.writer.add_image(image_path, image_show, (self.epoch - 1) * self.max_batches + self.batch)

		if (self.batch % self.max_batches) == 0:
			for loss_name, loss in self.losses.items():
				self.losses[loss_name] = 0.0
			self.epoch += 1
			self.batch = 1
			print('\n')
			
		else:
			self.batch += 1
			print('\n')
