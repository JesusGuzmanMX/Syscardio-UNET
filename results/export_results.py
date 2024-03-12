'''
Este modulo sirve para exportar los resultados obtenidos del modelo en los siguientes formatos y direcciones:

		---> Mascaras o segmentaciones en /results/masks

		---> Coordenadas de los contornos de las segmentaciones en /results/json_files

		---> Contornos sobrepuestos a sus respectivas imágenes de prueba en /results/contours


'''


import numpy as np 
import torch
import matplotlib.pyplot as plt

from utilities import ClassResultsProcessing, PostProcessing, SaveMasks, SaveCoordinates
import utilities.coordinates_functions as CoordinatesFunction


class ExportingResults:

	def __init__(self, all_pruebas, all_preds):


		p_array = np.zeros((4, 128, 128))
		c = 0

		# ciclo for para cada uno de los lotes
		for i in range(len(all_pruebas)):

			# lote actual i con forma [4 x N X 128 X 128]
			current_batch = all_pruebas[i]

			current_pred_batch = all_preds[i]


			# ciclo for para cada una de los elementos del lote actual
			for j in range(current_batch.shape[0]):
				
				# elemento actual j del lote actual i
				img = current_batch[j, 0]

				prediction = torch.argmax(current_pred_batch[j], dim = 0)

				current_volume_pred = ClassResultsProcessing.split_classes(prediction)

				# crear figura con la imagen j del lote i
				plt.imshow(img, cmap = 'gray')

				c = c + 1 

				cd = 0

				# ciclo for para cada clase 
				for k in range(current_volume_pred.shape[0]):

					p1 = current_volume_pred[k]

					pp = PostProcessing(p1)

					p = pp.post_img

					p_array[k,:,:] = p

					# exportar las predicciones
					SaveMasks(ClassResultsProcessing.merge_classes(p_array), c)

					# exportar coordenadas en archivo JSON
					cd = CoordinatesFunction.coordinates_dict(p, k, cd)
					SaveCoordinates(cd, c)

					pc = CoordinatesFunction.coordinates(p)

					plt.plot(pc[:,0], pc[:,1], 'g-')

					plt.xticks([])
					plt.yticks([])

				plt.savefig('./results/contours/' + str(c) + '.jpg')
				plt.clf()




