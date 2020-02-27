import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image

class ConvBlock(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(ConvBlock, self).__init__()

		self.layer = nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1),
		nn.BatchNorm2d(dim_out),
		nn.LeakyReLU(0.2, inplace=True))

	def forward(self, x):
		return self.layer(x)

class UpconvBlock(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(UpconvBlock, self).__init__()

		self.layer = nn.Sequential(nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1),
		nn.BatchNorm2d(dim_out),
		nn.ReLU(inplace=True))

	def forward(self, x):
		return self.layer(x)

class Generator(nn.Module):
	def __init__(self, ngf):
		super(Generator, self).__init__()
		self.ngf = ngf

		# Encoder

		self.rgb_e1 = nn.Sequential(nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1),
		nn.LeakyReLU(0.2, inplace=True))
		self.depth_e1 = nn.Sequential(nn.Conv2d(1, ngf, kernel_size=4, stride=2, padding=1),
		nn.LeakyReLU(0.2, inplace=True))

		self.rgb_e2 = ConvBlock(ngf, ngf * 2)
		self.depth_e2 = ConvBlock(ngf, ngf * 2)
		self.rgb_e3 = ConvBlock(ngf * 2, ngf * 4)
		self.depth_e3 = ConvBlock(ngf * 2, ngf * 4)
		self.rgb_e4 = ConvBlock(ngf * 4, ngf * 8)
		self.depth_e4 = ConvBlock(ngf * 4, ngf * 8)
		self.rgb_e5 = ConvBlock(ngf * 8, ngf * 8)
		self.depth_e5 = ConvBlock(ngf * 8, ngf * 8)
		self.rgb_e6 = ConvBlock(ngf * 8, ngf * 8)
		self.depth_e6 = ConvBlock(ngf * 8, ngf * 8)
		self.rgb_e7 = ConvBlock(ngf * 8, ngf * 8)
		self.depth_e7 = ConvBlock(ngf * 8, ngf * 8)
		self.rgb_e8 = ConvBlock(ngf * 8, ngf * 8)
		self.depth_e8 = ConvBlock(ngf * 8, ngf * 8)

		# Decoder

		self.rgb_d1 = UpconvBlock(ngf * 8, ngf * 8)
		self.depth_d1 = UpconvBlock(ngf * 8, ngf * 8)
		self.rgb_d2 = UpconvBlock(ngf * 16, ngf * 8)
		self.depth_d2 = UpconvBlock(ngf * 16, ngf * 8)
		self.rgb_d3 = UpconvBlock(ngf * 16, ngf * 8)
		self.depth_d3 = UpconvBlock(ngf * 16, ngf * 8)
		self.rgb_d4 = UpconvBlock(ngf * 16, ngf * 8)
		self.depth_d4 = UpconvBlock(ngf * 16, ngf * 8)
		self.rgb_d5 = UpconvBlock(ngf * 16, ngf * 4)
		self.depth_d5 = UpconvBlock(ngf * 16, ngf * 4)
		self.rgb_d6 = UpconvBlock(ngf * 8, ngf * 2)
		self.depth_d6 = UpconvBlock(ngf * 8, ngf * 2)
		self.rgb_d7 = UpconvBlock(ngf * 4, ngf)
		self.depth_d7 = UpconvBlock(ngf * 4, ngf)
		self.rgb_d8 = UpconvBlock(ngf * 2, ngf)
		self.depth_d8 = UpconvBlock(ngf * 2, ngf)

		# Head

		self.conv = nn.Sequential(nn.Conv2d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0),
		nn.BatchNorm2d(ngf),
		nn.ReLU(inplace=True),
		nn.Conv2d(ngf, 1, kernel_size=3, stride=1, padding=1),
		nn.Sigmoid())

	def forward(self, RGB, Depth):
		RGB_E1 = self.rgb_e1(RGB)
		Depth_E1 = self.depth_e1(Depth)
		RGB_E2 = self.rgb_e2(RGB_E1)
		Depth_E2 = self.depth_e2(Depth_E1)
		RGB_E3 = self.rgb_e3(RGB_E2)
		Depth_E3 = self.depth_e3(Depth_E2)
		RGB_E4 = self.rgb_e4(RGB_E3)
		Depth_E4 = self.depth_e4(Depth_E3)
		RGB_E5 = self.rgb_e5(RGB_E4)
		Depth_E5 = self.depth_e5(Depth_E4)
		RGB_E6 = self.rgb_e6(RGB_E5)
		Depth_E6 = self.depth_e6(Depth_E5)
		RGB_E7 = self.rgb_e7(RGB_E6)
		Depth_E7 = self.depth_e7(Depth_E6)
		RGB_E8 = self.rgb_e8(RGB_E7)
		Depth_E8 = self.depth_e8(Depth_E7)

		RGB_D = self.rgb_d1(RGB_E8)
		Depth_D = self.depth_d1(Depth_E8)

		RGB_D = torch.cat([RGB_D, RGB_E7], dim=1)
		Depth_D = torch.cat([Depth_D, Depth_E7], dim=1)
		RGB_D = self.rgb_d2(RGB_D)
		Depth_D = self.depth_d2(Depth_D)

		RGB_D = torch.cat([RGB_D, RGB_E6], dim=1)
		Depth_D = torch.cat([Depth_D, Depth_E6], dim=1)
		RGB_D = self.rgb_d3(RGB_D)
		Depth_D = self.depth_d3(Depth_D)

		RGB_D = torch.cat([RGB_D, RGB_E5], dim=1)
		Depth_D = torch.cat([Depth_D, Depth_E5], dim=1)
		RGB_D = self.rgb_d4(RGB_D)
		Depth_D = self.depth_d4(Depth_D)

		RGB_D = torch.cat([RGB_D, RGB_E4], dim=1)
		Depth_D = torch.cat([Depth_D, Depth_E4], dim=1)
		RGB_D = self.rgb_d5(RGB_D)
		Depth_D = self.depth_d5(Depth_D)

		RGB_D = torch.cat([RGB_D, RGB_E3], dim=1)
		Depth_D = torch.cat([Depth_D, Depth_E3], dim=1)
		RGB_D = self.rgb_d6(RGB_D)
		Depth_D = self.depth_d6(Depth_D)

		RGB_D = torch.cat([RGB_D, RGB_E2], dim=1)
		Depth_D = torch.cat([Depth_D, Depth_E2], dim=1)
		RGB_D = self.rgb_d7(RGB_D)
		Depth_D = self.depth_d7(Depth_D)

		RGB_D = torch.cat([RGB_D, RGB_E1], dim=1)
		Depth_D = torch.cat([Depth_D, Depth_E1], dim=1)
		RGB_D = self.rgb_d8(RGB_D)
		Depth_D = self.depth_d8(Depth_D)

		feat = torch.cat([RGB_D, Depth_D], dim=1)
		output = self.conv(feat)

		return output

class Discriminator(nn.Module):
	def __init__(self, ndf, n_layers=5):
		super(Discriminator, self).__init__()

		model = [nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1),
		nn.LeakyReLU(0.2, True)]

		out_dims = ndf
		for i in range (n_layers):
			in_dims = out_dims
			out_dims = ndf * min(2 ** (i + 1), 8)
			
			model += [nn.Conv2d(in_dims, out_dims, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2, True)]

		model += [nn.Conv2d(out_dims, 1, kernel_size=3, stride=1, padding=1)]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		output = self.model(x)
		return output

class SaliencyGAN(nn.Module):
	def __init__(self, args):
		super(SaliencyGAN, self).__init__()

		self.args = args
		self.device = torch.device('cuda' if args.cuda else 'cpu')
		self.generator = Generator(args.ngf)
		self.discriminator = Discriminator(args.ndf)

		self.generator.to(self.device)
		self.discriminator.to(self.device)
		
		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
				nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def load_weights(self, checkpoint, mode='train', pretrained=False):
		print("Load checkpoint from", checkpoint)
		netG_model_path = '{}/netG_model_epoch_{:02d}.pth'.format(self.args.checkpoint, self.args.n_epochs)
		netD_model_path = '{}/netD_model_epoch_{:02d}.pth'.format(self.args.checkpoint, self.args.n_epochs)
		if pretrained:
			netG_model_path = '{}/netG_model_pretrained.pth'.format(self.args.checkpoint)
		if mode == 'train':
			if os.path.exists(netG_model_path) and os.path.exists(netD_model_path):
				G_state = torch.load(netG_model_path)
				D_state = torch.load(netD_model_path)
				self.generator.load_state_dict(G_state, strict=False)
				self.discriminator.load_state_dict(D_state, strict=False)
			else:
				print('Check point is not exists...')
		else:
			if os.path.exists(netG_model_path):
				G_state = torch.load(netG_model_path)
				self.generator.load_state_dict(G_state, strict=False)
			else:
				print('Check point is not exists...')

	def save_weights(self, log_dir, epoch):
		netG_model = os.path.join(log_dir, 'netG_model_epoch_{:02d}.pth'.format(epoch))
		netD_model = os.path.join(log_dir, 'netD_model_epoch_{:02d}.pth'.format(epoch))
		torch.save(self.generator.state_dict(), netG_model)
		torch.save(self.discriminator.state_dict(), netD_model)

	def calc_gradient_penalty(self, real, fake):
		alpha = torch.rand(real.size(0), 1, 1, 1).to(self.device)
		x_hat = (alpha * real.data + (1 - alpha) * fake.data).requires_grad_(requires_grad=True)
		
		logits = self.discriminator(x_hat)
		grad_outputs = torch.ones(logits.size()).to(self.device)

		grads = torch.autograd.grad(outputs=logits,
									inputs=x_hat,
									grad_outputs=grad_outputs,
									create_graph=True,
									retain_graph=True,
									only_inputs=True)[0]
		grads = grads.view(logits.size(0), -1)
		grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1))
		
		output = torch.mean((grads_norm - 1) ** 2)
		return output

	def test_(self, dataloader):
		with torch.no_grad():
			for iteration, batch in enumerate(dataloader):
				rgb = batch['RGB'].to(self.device)
				depth = batch['Depth'].to(self.device)

				output = self.generator(rgb, depth).data

				save_image(output, self.args.output_dir + '/Saliency_map_%04d.png' % (iteration + 1))
				print('Saliency maps {:04d} / {:04d}'.format(iteration + 1, len(dataloader)))
			print('\n')

	def train_(self, dataloader, optimizerG, optimizerD, logger):
		device = torch.device('cuda' if self.args.cuda else 'cpu')
		for iteration, batch in enumerate(dataloader):
			rgb = batch['RGB'].to(self.device)
			depth = batch['Depth'].to(self.device)
			gt = batch['GT'].to(self.device)

			# Update Discriminator

			optimizerD.zero_grad()
		
			fake = self.generator(rgb, depth)
			pred_fake = self.discriminator(fake.detach())
			loss_d_fake = torch.mean(pred_fake)

			pred_real = self.discriminator(gt)
			loss_d_real = -torch.mean(pred_real)

			loss_d_gp = self.calc_gradient_penalty(gt, fake)

			loss_d = loss_d_fake + loss_d_real + self.args.lambda_gp * loss_d_gp

			loss_d.backward()
			optimizerD.step()

			# Update Generator

			optimizerG.zero_grad()

			fake = self.generator(rgb, depth)
			pred_fake = self.discriminator(fake)
			loss_g_fake = -torch.mean(pred_fake)

			loss_g_ce = F.binary_cross_entropy(fake, gt)

			loss_g = loss_g_fake + self.args.lambda_g * loss_g_ce

			loss_g.backward()
			optimizerG.step()

			logger.log(losses={'loss_d': loss_d, 'loss_d_fake': loss_d_fake, 'loss_d_real': loss_d_real, 'loss_d_gp': loss_d_gp,
								'loss_g': loss_g, 'loss_g_fake': loss_g_fake, 'loss_g_ce': loss_g_ce},
						images={'RGB': rgb, 'Depth': depth, 'GT': gt, 'Fake': fake})
