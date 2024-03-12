import torch
import numpy as np  


class ClassResultsProcessing:

	def split_classes(entrada):

		# tensor de tamaño [4 x 128 x 128]
		pred = torch.zeros(4, 128, 128)

		# tensor con las clases [BG, VI, AI, VD, AD]
		classes = [0, 1, 2, 3, 4]

		# ciclo for para cada cavidad [VI, AI, VD, AD]
		for i in range(1, len(classes)):
			pred[i-1,:,:] = (entrada == classes[i]).float()

		return pred

	def merge_classes(entrada):

		salida = np.where(entrada[1,:,:] == 1.0, 2, entrada[0,:,:])
		salida = np.where(entrada[2,:,:] == 1.0, 3, salida)
		salida = np.where(entrada[3,:,:] == 1.0, 4, salida)

		return salida
