'''
Este modulo es para realizar el proceso de testing del modelo con
	ejemplos nuevos o datos reales

'''


import torch


class Testing:

	def inferences(model, real_dl):

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		with torch.no_grad():

			# tensor vacío donde se guardaran las predicciones del modelo
			all_preds = []

			all_pruebas = []


			for real_examples in real_dl:

				prueba = real_examples
				prueba = prueba.type(torch.FloatTensor)

				prueba = prueba.to(device)


				# hacemos inferencia
				pred = model(prueba)


				prueba = prueba.to('cpu')
				pred = pred.to('cpu')

				#apilar ejemplos reales
				all_pruebas.append(prueba)

				#apilar inferencias hechas por el modelo
				all_preds.append(pred)

		return all_pruebas, all_preds


