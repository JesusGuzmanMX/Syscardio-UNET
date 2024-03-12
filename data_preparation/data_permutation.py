'''
Este modulo se encarga de realizar un desordenamiento aleatorio
	de las imagenes del dataset y sus respectivas mascaras,
	cuidando que sea el mismo desordenamiento para ambos y no se
	crucen mascaras que no correspondan a otra imagen.

'''


from data_preparation import ReadData

import numpy as np

class DataPermutation:


	def __init__(self):

		img_set, label_set = ReadData.create_dataset()

		# se crea una lista con 418 numeros desordenados
		permutation = np.random.permutation(img_set.shape[0])
		

		''' Se toma:
				-> 80% para entrenamiento
				-> 10% para validación
				-> 10% para pruebas
		'''

		trn_size = int(0.8*img_set.shape[0])
		val_size = int(0.1*img_set.shape[0])

		# train set
		self.trn_set = img_set[permutation[:trn_size]]
		self.trn_label = label_set[permutation[:trn_size]]

		# validation set
		self.val_set = img_set[permutation[trn_size:trn_size+val_size]]
		self.val_label = label_set[permutation[trn_size:trn_size+val_size]]

		# proof set
		self.p_set = img_set[permutation[trn_size+val_size:]]
		self.p_label = label_set[permutation[trn_size+val_size:]]



	def __getitem__(self):
		return self.trn_set, self.trn_label, self.val_set, self.val_label, self.p_set, self.p_label