from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import random

from torch.autograd import Variable
from torchvision.utils import save_image
from utils import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="Folder path which containing input images")
parser.add_argument("--mode", required=True, choices=["train", "test"], help="Train or test mode selection")
parser.add_argument("--output_dir", required=True, help="Folder path to save output images")
parser.add_argument("--tensorboard_dir", required=False, help="Folder path to save tensorboard logs")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="Folder path to resume training from or use for testing")
parser.add_argument("--pretrained", action='store_true', help="Using pretrained model")
parser.add_argument("--n_epochs", type=int, default=0, help="Load checkpoint from trained models with n_epochs")
parser.add_argument("--max_epochs", type=int, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Number of images in batch")
parser.add_argument("--cuda", action='store_true', help="Using cuda")
parser.add_argument("--threds", type=int, default=4, help="Number of threds for data loading")
parser.add_argument("--ngf", type=int, default=64, help="Number of generators filters in first convolution layer")
parser.add_argument("--ndf", type=int, default=16, help="Number of discriminator filters in first convolution layer")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer")
parser.add_argument("--beta1", type=float, default=0.9, help="Momentum of Adam optimizer")
parser.add_argument("--ce_weight", type=float, default=10.0, help="Weight on CE term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="Weight on GAN term for generator gradient")
args = parser.parse_args()

if args.cuda and not torch.cuda.is_available():
	raise Exception("Can't run with GPU, please run without --cuda")

if args.seed is None:
	args.seed = random.randint(0, 2 ** 31 - 1)

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

if args.mode == "test":
	args.batch_size = 1

dataloader = load_images(args)
device = torch.device('cuda:0' if args.cuda else 'cpu')

if args.mode == "train":
	netG = create_generator(3, 1, args.ngf)
	netD = create_discriminator(4, args.ndf)

	criterionGAN = GANLoss()
	criterionCE = AdaptiveCELoss()

	optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
	optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

	if args.cuda:
		netG = netG.cuda()
		netD = netD.cuda()
		criterionGAN = criterionGAN.cuda()
		criterionCE = criterionCE.cuda()

	if args.checkpoint is not None:
		netG_model_path = '{}/netG_model_epoch_{}.pth'.format(args.checkpoint, args.n_epochs)
		netD_model_path = '{}/netD_model_epoch_{}.pth'.format(args.checkpoint, args.n_epochs)
		netG.load_state_dict(torch.load(netG_model_path))
		netD.load_state_dict(torch.load(netD_model_path))

	logger = Logger(args.n_epochs + 1, args.max_epochs, len(dataloader), args.tensorboard_dir)

	for epoch in range(args.n_epochs + 1, args.max_epochs + 1):
		for iteration, batch in enumerate(dataloader):
			RGB = batch['RGB'].to(device)
			Depth = batch['Depth'].to(device)
			GT = batch['GT'].to(device)
			fake = netG(RGB, Depth)

			# Update Discriminator

			optimizerD.zero_grad()

			pred_fake = netD(RGB.detach(), fake.detach())
			loss_d_fake = criterionGAN(pred_fake, False)

			pred_real = netD(RGB, GT)
			loss_d_real = criterionGAN(pred_real, True)

			loss_d = (loss_d_fake + loss_d_real) * 0.5

			loss_d.backward()

			optimizerD.step()

			# Update Generator

			optimizerG.zero_grad()

			pred_fake = netD(RGB, fake)
			loss_g_gan = criterionGAN(pred_fake, True)

			loss_g_ce = criterionCE(fake, GT)

			loss_g = args.gan_weight * loss_g_gan + args.ce_weight * loss_g_ce

			loss_g.backward()

			optimizerG.step()

			logger.log(losses = {'loss_g': loss_g, 'loss_g_ce': loss_g_ce, 'loss_g_gan': loss_g_gan, 'loss_d': loss_d},
						images = {'RGB': RGB, 'Depth': Depth, 'GT': GT, 'fake': fake})

		if epoch % 10 == 0:
			netG_model_output_path = '{}/netG_model_epoch_{}.pth'.format(args.output_dir, epoch)
			netD_model_output_path = '{}/netD_model_epoch_{}.pth'.format(args.output_dir, epoch)
			torch.save(netG.state_dict(), netG_model_output_path)
			torch.save(netD.state_dict(), netD_model_output_path)

else:
	netG = create_generator(3, 1, args.ngf)
	if args.cuda:
		netG = netG.cuda()

	if args.pretrained:
		netG_model_path = '{}/netG_model_pretrained.pth'.format(args.checkpoint)
	else:
		netG_model_path = '{}/netG_model_epoch_{}.pth'.format(args.checkpoint, args.n_epochs)
	netG.load_state_dict(torch.load(netG_model_path))
	netG.eval()

	for iteration, batch in enumerate(dataloader):
		RGB = batch['RGB'].to(device)
		Depth = batch['Depth'].to(device)

		fake = netG(RGB, Depth).data

		save_image(fake, args.output_dir + '/Test_%04d-outputs.png' % (iteration + 1))
		print('Generated images {:04d} of {:04d}'.format(iteration + 1, len(dataloader)))
		
	print('\n')
