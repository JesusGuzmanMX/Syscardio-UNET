from torchsummary import summary
import warnings

import torch

from training import TrainingFunction, LoadModel
from data_preparation import BatchDataProcessing
from testing import Testing
from results import ExportingResults





print("train?:")
parameter = input('[Y/N]: ')
print('\n')

while True:

	if parameter == 'Y':

		print('Loading training dataset...\n')

		trn_dl, val_dl = BatchDataProcessing.batching()

		print('DATASET SUCCESSFULLY LOADED!\n')



		print('Loading model...\n')

		UNet = LoadModel()

		print('MODEL SUCCESSFULLY LAODED!\n')

		warnings.filterwarnings("ignore")



		print("Starting the training!\n")

		model, model_weights = TrainingFunction.train(UNet.model, 
													trn_dl, val_dl,
													learning_rate = 1e-3, 
													EPOCHS = 50)

		print('TRAINING DONE!\n')
		print("MODEL SAVED AS: model.pt !\n")
		parameter = 'N'


	elif parameter == 'N':

		print("testing?:")

		parameter = input('[Y/N]: ')

		print('\n')



		if parameter == 'Y':

			print('Loading testing dataset...\n')

			real_dl = BatchDataProcessing.batching_real()

			print('DATASET SUCCESSFULLY LOADED!\n')



			print('Loading trained model...\n')

			UNet = torch.load('model.pt')

			print('MODEL SUCCESSFULLY LAODED!\n')



			print('Model making inference...\n')

			all_pruebas, all_preds = Testing.inferences(UNet, real_dl)

			print('INFERENCE DONE!\n')



			print("Exporting results...\n")

			ExportingResults(all_pruebas, all_preds)

			print("RESULTS EXPORTED AT results DIR!")
			print('\n')
			print('Hasta luego!')

			break

		else:
			print('Hasta luego!')
			break

	else:

		print('Hasta luego!')

		break




