'''
Este modulo convierte los volumenes tanto de entrenamiento como de pruebas
	en numpy array en 8 bits para facilitar su manejo a lo largo del 
	codigo.

'''


from PIL import Image
import numpy as np 


from dataset import Dataset 

class ReadData:

	def create_dataset():

		img_path = Dataset.files_path('img')
		label_path = Dataset.files_path('label')
		
		# se leen ambos volumenes
		img = Image.open(img_path)
		label = Image.open(label_path)

		# extraemos el alto y ancho de las imagenes 
		height, width = np.shape(img)

		# crear una matriz vacía de tamaño (418 x 112 x 112)
		img_np_array = np.zeros((img.n_frames, height, width))
		label_np_array = np.zeros((label.n_frames, height, width))

		for i in range(img.n_frames):
			img.seek(i)
			img_np_array[i,:,:] = np.array(img)

		for i in range(label.n_frames):
			label.seek(i)
			label_np_array[i,:,:] = np.array(label)


		return img_np_array.astype('uint8'), label_np_array.astype('uint8')

	def create_real_dataset():

		real_path = Dataset.files_path('real')

		real_img = Image.open(real_path)

		height, width = np.shape(real_img)

		real_img_np_array = np.zeros((real_img.n_frames, height, width))

		for i in range(real_img.n_frames):
			real_img.seek(i)
			real_img_np_array[i,:,:] = np.array(real_img)

		return real_img_np_array

