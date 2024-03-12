'''
batching: Esta funcion se encarga de organizar los datos de entrenamiento y
			validacion en lotes que sirvan de entrada para el modelo

batching_real: Al igual que con la funcion de "batching" se encarga de organizar 
			los datos reales o de testing en lotes que sirvan de entrada para
			el modelo
'''

from data_preparation import DataPermutation, Transformations, DataPreprocessing, RealDataPreprocessing, ReadData

import torchvision.transforms as T
from torch.utils.data import DataLoader

class BatchDataProcessing:

	def batching():

		# --------  Parámetros -----------
		#								#
		# Tamaño de lote				#
		tam_lot = 8						#
		#								#
		# Desordenar?					#
		desorden = True					#
		#								#
		# Número de procesos paralelos	#
		procesos = 0					#
		#								#
		# --------------------------------

		sets = DataPermutation()

		trn_set = sets.trn_set
		trn_label = sets.trn_label
		val_set = sets.val_set
		val_label = sets.val_label


		t = Transformations()


		trn_ds = DataPreprocessing(trn_set, trn_label, t.tsfm, t.tsfm)
		val_ds = DataPreprocessing(val_set, val_label, t.tsfm, t.tsfm)

		# Creación de DataLoaders (train and validation)
		trn_dl = DataLoader(trn_ds, batch_size = tam_lot, 
									shuffle = desorden, 
									num_workers = procesos)
		val_dl = DataLoader(val_ds, batch_size = 4, 
									shuffle = desorden, 
									num_workers = procesos)


		return trn_dl, val_dl

		

	def batching_real():

		real_set = ReadData.create_real_dataset()

		t = Transformations()

		real_ds = RealDataPreprocessing(real_set, t.tsfm)

		real_dl = DataLoader(real_ds, batch_size = 4)

		return real_dl


