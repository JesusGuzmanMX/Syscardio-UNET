'''
Este modulo carga la UNet y la carga en GPU o CPU

'''


from model import UNet

import torch

class LoadModel:

	def __init__(self):

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model = UNet()


	def __getitem__(self):
		if torch.cuda.is_available():
			
			return self.model.cuda(), self.device