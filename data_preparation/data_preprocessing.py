'''
Este modulo realiza un ligero preprocesamiento en las imágenes 
	utilizando las transformaciones indicadas preparando los datos 
	para el modelo y su entrenamiento.

'''


from PIL import Image 
import numpy as np 
import torchvision.transforms as T 
import torchvision.transforms.functional as TF 
import random

from data_preparation import DataNormalization


class DataPreprocessing:

	def __init__(self, images, labels, tsfm_images = None, tsfm_labels = None, DA = False):

		# atributos
		self.labels = labels
		self.tsfm_images = tsfm_images
		self.tsfm_labels = tsfm_labels
		self.images = images
		self.DA = DA

	def __getitem__(self,i):

		x = self.images[i]
		y = self.labels[i]

		y = y.astype('float')

		x = Image.fromarray(x)
		y = Image.fromarray(y)


		# transformaciones brindadas por torchvision
		if self.tsfm_images is not None:
			x = self.tsfm_images(x)
		if self.tsfm_labels is not None:
			y = self.tsfm_labels(y)
		if self.DA == True:
			a, b, c, d = T.RandomAffine.get_params(degrees = (0,0), translate = [0.2, 0.2],
				scale_ranges = None, shears = None, img_size = (128,128))

			# Random Shift

			if random.random() > 0.5:
				x = TF.affine(x, a, b, c, d)
				y = TF.affine(y, a, b, c, d)

		x = DataNormalization.normalization(x)


		return x, y


	def __len__(self):
		return len(self.images)