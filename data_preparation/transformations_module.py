'''
Este modulo indica las transformaciones utilizadas en el codigo:
		-> Redimension de imagenes [128 x 128]
'''

import torchvision.transforms as T  

class Transformations:

	def __init__(self):

		self.tsfm = T.Compose([

			# redimension
			T.Resize(128),

			# convertir a tensor
			T.ToTensor()

			])

	def __getitem__(self):
		return self.tsfm