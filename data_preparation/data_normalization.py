'''
Este modulo se encarga de realizar una normalizacion en [0 a 1] en
	los valores de los pixeles de las imagenes.
'''


class DataNormalization:


	def normalization(input_data):

		input_size = input_data.shape
		x = input_data.view(input_size[0], -1).float()
		x -= x.min(1, keepdim = True)[0]
		x /= x.max(1, keepdim = True)[0]
		normalized_data = x.view(input_size).float()

		return normalized_data