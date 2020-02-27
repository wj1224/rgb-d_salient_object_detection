import torch
import torch.nn as nn
import argparse
import os
import random

from utils import load_images, Logger
from models import SaliencyGAN

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", help="Folder path which containing input images")
	parser.add_argument("--mode", required=True, choices=["train", "test"], help="Mode selection")
	parser.add_argument("--output_dir", required=True, help="Folder path to save the model weights or generated images")
	parser.add_argument("--seed", type=int, default=333)
	parser.add_argument("--checkpoint", default=None, help="Folder path to resume training")
	parser.add_argument("--n_epochs", type=int, default=0, help="Load checkpoint from trained models with n_epochs")
	parser.add_argument("--max_epochs", type=int, help="Number of training epochs")
	parser.add_argument("--pretrained", action='store_true', help="Using pretrained model in test mode")
	parser.add_argument("--batch_size", type=int, default=16, help="Number of images in batch")
	parser.add_argument("--cuda", action='store_true', help="Using cuda")
	parser.add_argument("--threds", type=int, default=4, help="Number of threds for data loading")
	parser.add_argument("--ngf", type=int, default=64, help="Number of generators filters in first convolution layer")
	parser.add_argument("--ndf", type=int, default=16, help="Number of discriminator filters in first convolution layer")
	parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer")
	parser.add_argument("--beta1", type=float, default=0.9, help="Momentum of Adam optimizer")
	parser.add_argument("--lambda_g", type=float, default=10.0, help="Weight on CE term for generator gradient")
	parser.add_argument("--lambda_gp", type=float, default=10.0, help="Weight on gradient penalty term of WGAN-GP loss")
	args = parser.parse_args()

	if args.cuda and not torch.cuda.is_available():
		raise Exception("Can't run with GPU, please run without --cuda")

	if args.seed is None:
		args.seed = random.randint(0, 2 ** 31 - 1)

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
	random.seed(args.seed)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	device = torch.device('cuda' if args.cuda else 'cpu')

	if args.mode == 'train':
		dataloader = load_images(args)

		model = SaliencyGAN(args).to(device)

		optimizerG = torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
		optimizerD = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

		model.load_weights(args.checkpoint, args.mode, args.pretrained)

		logger = Logger(args.n_epochs + 1, args.max_epochs, len(dataloader), args.output_dir)

		for epoch in range(args.n_epochs + 1, args.max_epochs + 1):
			model.train()
			model.train_(dataloader, optimizerG, optimizerD, logger)
			model.save_weights(args.output_dir, epoch)

	else:
		dataloader = load_images(args)

		model = SaliencyGAN(args).to(device)
		model.load_weights(args.checkpoint, args.mode, args.pretrained)

		model.eval()
		model.test_(dataloader)
