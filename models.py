from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from torch.autograd import Variable

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def create_generator(input_nc, output_nc, ngf):
	netG = Generator(input_nc, output_nc, ngf)
	netG.apply(weights_init)
	return netG

def create_discriminator(input_nc, ndf):
	netD = Discriminator(input_nc, ndf)
	netD.apply(weights_init)
	return netD

class Generator(nn.Module):
	def __init__(self, input_nc, output_nc, ngf):
		super(Generator, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf

		# Encoder

		self.encoder1_1_conv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
		self.encoder1_1_lrelu = nn.LeakyReLU(0.2, True)
		self.encoder2_1_conv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
		self.encoder2_1_lrelu = nn.LeakyReLU(0.2, True)

		self.encoder1_2_conv = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)
		self.encoder1_2_norm = nn.BatchNorm2d(ngf * 2)
		self.encoder1_2_lrelu = nn.LeakyReLU(0.2, True)
		self.encoder2_2_conv = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)
		self.encoder2_2_norm = nn.BatchNorm2d(ngf * 2)
		self.encoder2_2_lrelu = nn.LeakyReLU(0.2, True)

		self.encoder1_3_conv = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1)
		self.encoder1_3_norm = nn.BatchNorm2d(ngf * 4)
		self.encoder1_3_lrelu = nn.LeakyReLU(0.2, True)
		self.encoder2_3_conv = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1)
		self.encoder2_3_norm = nn.BatchNorm2d(ngf * 4)
		self.encoder2_3_lrelu = nn.LeakyReLU(0.2, True)

		self.encoder1_4_conv = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder1_4_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder1_4_lrelu = nn.LeakyReLU(0.2, True)
		self.encoder2_4_conv = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder2_4_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder2_4_lrelu = nn.LeakyReLU(0.2, True)

		self.encoder1_5_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder1_5_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder1_5_lrelu = nn.LeakyReLU(0.2, True)
		self.encoder2_5_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder2_5_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder2_5_lrelu = nn.LeakyReLU(0.2, True)

		self.encoder1_6_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder1_6_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder1_6_lrelu = nn.LeakyReLU(0.2, True)
		self.encoder2_6_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder2_6_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder2_6_lrelu = nn.LeakyReLU(0.2, True)

		self.encoder1_7_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder1_7_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder1_7_lrelu = nn.LeakyReLU(0.2, True)
		self.encoder2_7_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder2_7_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder2_7_lrelu = nn.LeakyReLU(0.2, True)

		self.encoder1_8_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder1_8_norm = nn.BatchNorm2d(ngf * 8)
		self.encoder2_8_conv = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.encoder2_8_norm = nn.BatchNorm2d(ngf * 8)

		# Decoder

		self.decoder1_8_relu = nn.ReLU(True)
		self.decoder1_8_deconv = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder1_8_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder2_8_relu = nn.ReLU(True)
		self.decoder2_8_deconv = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder2_8_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder1_7_relu = nn.ReLU(True)
		self.decoder1_7_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder1_7_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder2_7_relu = nn.ReLU(True)
		self.decoder2_7_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder2_7_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder1_6_relu = nn.ReLU(True)
		self.decoder1_6_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder1_6_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder2_6_relu = nn.ReLU(True)
		self.decoder2_6_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder2_6_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder1_5_relu = nn.ReLU(True)
		self.decoder1_5_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder1_5_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder2_5_relu = nn.ReLU(True)
		self.decoder2_5_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1)
		self.decoder2_5_norm = nn.BatchNorm2d(ngf * 8)

		self.decoder1_4_relu = nn.ReLU(True)
		self.decoder1_4_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1)
		self.decoder1_4_norm = nn.BatchNorm2d(ngf * 4)

		self.decoder2_4_relu = nn.ReLU(True)
		self.decoder2_4_deconv = nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1)
		self.decoder2_4_norm = nn.BatchNorm2d(ngf * 4)

		self.decoder1_3_relu = nn.ReLU(True)
		self.decoder1_3_deconv = nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1)
		self.decoder1_3_norm = nn.BatchNorm2d(ngf * 2)

		self.decoder2_3_relu = nn.ReLU(True)
		self.decoder2_3_deconv = nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1)
		self.decoder2_3_norm = nn.BatchNorm2d(ngf * 2)

		self.decoder1_2_relu = nn.ReLU(True)
		self.decoder1_2_deconv = nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1)
		self.decoder1_2_norm = nn.BatchNorm2d(ngf)

		self.decoder2_2_relu = nn.ReLU(True)
		self.decoder2_2_deconv = nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1)
		self.decoder2_2_norm = nn.BatchNorm2d(ngf)

		self.decoder1_1_relu = nn.ReLU(True)
		self.decoder1_1_deconv = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1)
		self.decoder1_1_norm = nn.BatchNorm2d(ngf)
		self.decoder2_1_relu = nn.ReLU(True)
		self.decoder2_1_deconv = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1)
		self.decoder2_1_norm = nn.BatchNorm2d(ngf)

		self.decoder1_0_relu = nn.ReLU(True)
		self.decoder1_0_conv = nn.Conv2d(ngf, output_nc, kernel_size=1)
		self.decoder1_0_norm = nn.BatchNorm2d(1)
		self.decoder2_0_relu = nn.ReLU(True)
		self.decoder2_0_conv = nn.Conv2d(ngf, output_nc, kernel_size=1)
		self.decoder2_0_norm = nn.BatchNorm2d(1)

		self.decoder0 = nn.Sigmoid()

	def forward(self, RGB, depth):
		RGB_conv1 = self.encoder1_1_conv(RGB)
		Depth_conv1 = self.encoder2_1_conv(depth)

		RGB_conv2 = self.encoder1_1_lrelu(RGB_conv1)
		RGB_conv2 = self.encoder1_2_conv(RGB_conv2)
		RGB_conv2 = self.encoder1_2_norm(RGB_conv2)
		Depth_conv2 = self.encoder2_1_lrelu(Depth_conv1)
		Depth_conv2 = self.encoder2_2_conv(Depth_conv2)
		Depth_conv2 = self.encoder2_2_norm(Depth_conv2)

		RGB_conv3 = self.encoder1_2_lrelu(RGB_conv2)
		RGB_conv3 = self.encoder1_3_conv(RGB_conv3)
		RGB_conv3 = self.encoder1_3_norm(RGB_conv3)
		Depth_conv3 = self.encoder2_2_lrelu(Depth_conv2)
		Depth_conv3 = self.encoder2_3_conv(Depth_conv3)
		Depth_conv3 = self.encoder2_3_norm(Depth_conv3)

		RGB_conv4 = self.encoder1_3_lrelu(RGB_conv3)
		RGB_conv4 = self.encoder1_4_conv(RGB_conv4)
		RGB_conv4 = self.encoder1_4_norm(RGB_conv4)
		Depth_conv4 = self.encoder2_3_lrelu(Depth_conv3)
		Depth_conv4 = self.encoder2_4_conv(Depth_conv4)
		Depth_conv4 = self.encoder2_4_norm(Depth_conv4)

		RGB_conv5 = self.encoder1_4_lrelu(RGB_conv4)
		RGB_conv5 = self.encoder1_5_conv(RGB_conv5)
		RGB_conv5 = self.encoder1_5_norm(RGB_conv5)
		Depth_conv5 = self.encoder2_4_lrelu(Depth_conv4)
		Depth_conv5 = self.encoder2_5_conv(Depth_conv5)
		Depth_conv5 = self.encoder2_5_norm(Depth_conv5)

		RGB_conv6 = self.encoder1_5_lrelu(RGB_conv5)
		RGB_conv6 = self.encoder1_6_conv(RGB_conv6)
		RGB_conv6 = self.encoder1_6_norm(RGB_conv6)
		Depth_conv6 = self.encoder2_5_lrelu(Depth_conv5)
		Depth_conv6 = self.encoder2_6_conv(Depth_conv6)
		Depth_conv6 = self.encoder2_6_norm(Depth_conv6)

		RGB_conv7 = self.encoder1_6_lrelu(RGB_conv6)
		RGB_conv7 = self.encoder1_7_conv(RGB_conv7)
		RGB_conv7 = self.encoder1_7_norm(RGB_conv7)
		Depth_conv7 = self.encoder2_6_lrelu(Depth_conv6)
		Depth_conv7 = self.encoder2_7_conv(Depth_conv7)
		Depth_conv7 = self.encoder2_7_norm(Depth_conv7)

		RGB_conv8 = self.encoder1_7_lrelu(RGB_conv7)
		RGB_conv8 = self.encoder1_8_conv(RGB_conv8)
		RGB_conv8 = self.encoder1_8_norm(RGB_conv8)
		Depth_conv8 = self.encoder2_7_lrelu(Depth_conv7)
		Depth_conv8 = self.encoder2_8_conv(Depth_conv8)
		Depth_conv8 = self.encoder2_8_norm(Depth_conv8)

		RGB_deconv8 = self.decoder1_8_relu(RGB_conv8)
		RGB_deconv8 = self.decoder1_8_deconv(RGB_deconv8)
		RGB_deconv8 = self.decoder1_8_norm(RGB_deconv8)

		Depth_deconv8 = self.decoder2_8_relu(Depth_conv8)
		Depth_deconv8 = self.decoder2_8_deconv(Depth_deconv8)
		Depth_deconv8 = self.decoder2_8_norm(Depth_deconv8)

		RGB_deconv7 = torch.cat([RGB_deconv8, RGB_conv7], dim=1)
		RGB_deconv7 = self.decoder1_7_relu(RGB_deconv7)
		RGB_deconv7 = self.decoder1_7_deconv(RGB_deconv7)
		RGB_deconv7 = self.decoder1_7_norm(RGB_deconv7)

		Depth_deconv7 = torch.cat([Depth_deconv8, Depth_conv7], dim=1)
		Depth_deconv7 = self.decoder2_7_relu(Depth_deconv7)
		Depth_deconv7 = self.decoder2_7_deconv(Depth_deconv7)
		Depth_deconv7 = self.decoder2_7_norm(Depth_deconv7)

		RGB_deconv6 = torch.cat([RGB_deconv7, RGB_conv6], dim=1)
		RGB_deconv6 = self.decoder1_6_relu(RGB_deconv6)
		RGB_deconv6 = self.decoder1_6_deconv(RGB_deconv6)
		RGB_deconv6 = self.decoder1_6_norm(RGB_deconv6)

		Depth_deconv6 = torch.cat([Depth_deconv7, Depth_conv6], dim=1)
		Depth_deconv6 = self.decoder2_6_relu(Depth_deconv6)
		Depth_deconv6 = self.decoder2_6_deconv(Depth_deconv6)
		Depth_deconv6 = self.decoder2_6_norm(Depth_deconv6)

		RGB_deconv5 = torch.cat([RGB_deconv6, RGB_conv5], dim=1)
		RGB_deconv5 = self.decoder1_5_relu(RGB_deconv5)
		RGB_deconv5 = self.decoder1_5_deconv(RGB_deconv5)
		RGB_deconv5 = self.decoder1_5_norm(RGB_deconv5)

		Depth_deconv5 = torch.cat([Depth_deconv6, Depth_conv5], dim=1)
		Depth_deconv5 = self.decoder2_5_relu(Depth_deconv5)
		Depth_deconv5 = self.decoder2_5_deconv(Depth_deconv5)
		Depth_deconv5 = self.decoder2_5_norm(Depth_deconv5)

		RGB_deconv4 = torch.cat([RGB_deconv5, RGB_conv4], dim=1)
		RGB_deconv4 = self.decoder1_4_relu(RGB_deconv4)
		RGB_deconv4 = self.decoder1_4_deconv(RGB_deconv4)
		RGB_deconv4 = self.decoder1_4_norm(RGB_deconv4)

		Depth_deconv4 = torch.cat([Depth_deconv5, Depth_conv4], dim=1)
		Depth_deconv4 = self.decoder2_4_relu(Depth_deconv4)
		Depth_deconv4 = self.decoder2_4_deconv(Depth_deconv4)
		Depth_deconv4 = self.decoder2_4_norm(Depth_deconv4)

		RGB_deconv3 = torch.cat([RGB_deconv4, RGB_conv3], dim=1)
		RGB_deconv3 = self.decoder1_3_relu(RGB_deconv3)
		RGB_deconv3 = self.decoder1_3_deconv(RGB_deconv3)
		RGB_deconv3 = self.decoder1_3_norm(RGB_deconv3)

		Depth_deconv3 = torch.cat([Depth_deconv4, Depth_conv3], dim=1)
		Depth_deconv3 = self.decoder2_3_relu(Depth_deconv3)
		Depth_deconv3 = self.decoder2_3_deconv(Depth_deconv3)
		Depth_deconv3 = self.decoder2_3_norm(Depth_deconv3)

		RGB_deconv2 = torch.cat([RGB_deconv3, RGB_conv2], dim=1)
		RGB_deconv2 = self.decoder1_2_relu(RGB_deconv2)
		RGB_deconv2 = self.decoder1_2_deconv(RGB_deconv2)
		RGB_deconv2 = self.decoder1_2_norm(RGB_deconv2)

		Depth_deconv2 = torch.cat([Depth_deconv3, Depth_conv2], dim=1)
		Depth_deconv2 = self.decoder2_2_relu(Depth_deconv2)
		Depth_deconv2 = self.decoder2_2_deconv(Depth_deconv2)
		Depth_deconv2 = self.decoder2_2_norm(Depth_deconv2)

		RGB_deconv1 = torch.cat([RGB_deconv2, RGB_conv1], dim=1)
		RGB_deconv1 = self.decoder1_1_relu(RGB_deconv1)
		RGB_deconv1 = self.decoder1_1_deconv(RGB_deconv1)
		RGB_deconv1 = self.decoder1_1_norm(RGB_deconv1)

		Depth_deconv1 = torch.cat([Depth_deconv2, Depth_conv1], dim=1)
		Depth_deconv1 = self.decoder2_1_relu(Depth_deconv1)
		Depth_deconv1 = self.decoder2_1_deconv(Depth_deconv1)
		Depth_deconv1 = self.decoder2_1_norm(Depth_deconv1)

		RGB_deconv0 = self.decoder1_0_relu(RGB_deconv1)
		RGB_deconv0 = self.decoder1_0_conv(RGB_deconv0)
		RGB_deconv0 = self.decoder1_0_norm(RGB_deconv0)
		Depth_deconv0 = self.decoder2_0_relu(Depth_deconv1)
		Depth_deconv0 = self.decoder2_0_conv(Depth_deconv0)
		Depth_deconv0 = self.decoder2_0_norm(Depth_deconv0)

		output = torch.add(RGB_deconv0, Depth_deconv0)
		output = self.decoder0(output)

		return output

class Discriminator(nn.Module):
	def __init__(self, input_nc, ndf, n_layers=3):
		super(Discriminator, self).__init__()

		model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
		nn.LeakyReLU(0.2, True)]

		out_dims = ndf
		for i in range (n_layers):
			in_dims = out_dims
			out_dims = ndf * min(2 ** (i + 1), 8)
			if i == n_layers - 1:
				stride = 1
			else:
				stride = 2
			model += [nn.Conv2d(in_dims, out_dims, kernel_size=4, stride=stride, padding=1),
			nn.BatchNorm2d(out_dims),
			nn.LeakyReLU(0.2, True)]

		model += [nn.Conv2d(ndf * min(2 ** n_layers, 8), 1, kernel_size=4, stride=1, padding=1),
		nn.Sigmoid()]

		self.model = nn.Sequential(*model)

	def forward(self, input1, input2):
		input = torch.cat((input1, input2), dim=1)
		output = self.model(input[:,:4,:,:])
		return output

class GANLoss(nn.Module):
	def __init__(self, use_lsgan = False, target_real_label=1.0, target_fake_label=0.0,
				 tensor = torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor.cuda())

class AdaptiveCELoss(nn.Module):
	def __init__(self):
		super(AdaptiveCELoss, self).__init__()

	def forward(self, input, target):
		b, _, w, h = target.size()
		tsize = b * w * h
		EPS  = 1e-12
		beta_i = torch.sum(target[:,0,:,:]).item() / tsize
		loss = -(((1 - beta_i) * (input[:,0,:,:] * (torch.log(target[:,0,:,:] + EPS)))) + ((beta_i) * ((1 - input[:,0,:,:]) * torch.log(torch.abs(1 - target[:,0,:,:] + EPS)))))
		loss = torch.mean(loss)

		return loss
