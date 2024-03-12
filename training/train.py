'''
Este modulo es para realizar el entrenamiento de la UNet utilizando
	los datos de entrenamiento y validacion. Tambien guarda el modelo y sus pesos 
	como un archivo: model.pt

'''


from training import LoadModel
import utilities.dice_coeff as DiceFunction
import utilities.iou as IoUFunction
import utilities.pixel_acc as PixelFunction

from tqdm import tqdm
import torch.optim as optim
import torch
from torch import nn
import copy 
import numpy as np
import matplotlib.pyplot as plt


class TrainingFunction:
	
	def train(model, trn_dl, val_dl, learning_rate, EPOCHS = 100):

		# -------------- inicializacion metricas ---------------
		dice_score = 0 
		iou_score = 0 
		pixel_score = 0


		best_dice = 0
		best_iou = 0
		best_pixel = 0


		# historicos de metricas
		hist_epoch = []
		hist_dice_coeff = []
		hist_iou = []
		hist_pixel = []
		hist_trn_loss = []
		hist_val_loss = []

		# ---------------------------------------------------

		# optimizador
		opt = optim.Adam(model.parameters(), lr = learning_rate)

		# función de pérdida
		loss_fn = nn.CrossEntropyLoss()

		scaler = torch.cuda.amp.GradScaler()

		best_model_wts = copy.deepcopy(model.state_dict())

		d = LoadModel()
		device = d.device


		# Ciclos de entrenamiento
		for epoch in range(EPOCHS):

			# imprimir la época actual
			print('epoch: ', epoch+1)

			losses = []

			# modelo en modo entrenamiento
			model.train()

			for x, y_true in trn_dl:

				# cargar lotes a gpu
				x = x.to(device)
				y_true = y_true.long().squeeze(1).to(device)

				with torch.cuda.amp.autocast():
					preds = model(x)

					# medir perdida
					loss = loss_fn(preds, y_true)

				# vaciamos los gradientes
				opt.zero_grad()
				scaler.scale(loss).backward()
				scaler.step(opt)
				scaler.update()

				losses.append(loss.item())

			trn_loss = np.mean(losses)

			#imprimir perdida de la epoca
			print('trn_loss: ', trn_loss)

			#guardar perdida en historico
			hist_trn_loss.append(trn_loss)



			#desactivamos temporalmente la grafica de computo
			with torch.no_grad():

				#modelo en modo de evaluacion
				model.eval()

				dices = []
				ious = []
				pixels = []
				val_losses = []

				#validacion de la epoca
				for x, y in val_dl:
					x = x.to(device)
					y = y.long().squeeze(1).to(device)

					#hacemos inferencia para obtener logits
					preds = torch.sigmoid(model(x))

					#obtener perdida para validacion
					v_loss = loss_fn(model(x), y)

					#sacara metricas
					dice_score = DiceFunction.dice_batch(preds, y) #<---------
					iou_score = IoUFunction.iou_batch(preds, y)
					pixel_score = PixelFunction.pixel_batch(preds, y)

					val_losses.append(v_loss.item())


				dices.append(dice_score.item())
				ious.append(iou_score.item())
				pixels.append(pixel_score.item())


			val_loss = np.mean(val_losses)
			print('val_loss: ', val_loss)


			dice = np.mean(dices)
			iou = np.mean(ious)
			pixel = np.mean(pixels)


			#historico de perdidas en validacion
			hist_val_loss.append(val_loss)

			#actualizacion historico de metricas
			hist_dice_coeff.append(dice)
			hist_iou.append(iou)
			hist_pixel.append(pixel)
			hist_epoch.append(epoch)


			#imprimir metricas
			print('\n')
			print(f'dice_score: {dice: .2f}\n')
			print(f'iou_score: {iou: .2f}\n')
			print(f'pixel_score: {pixel: .2f}')
			print('\n')


			if dice > best_dice:
				best_dice = dice

			if iou > best_iou:
				best_iou = iou 

			if pixel > best_pixel:
				best_pixel = pixel 


			#guardar los pesos del mejor coeficiente de Dice
			best_model_wts = copy.deepcopy(model.state_dict())


		#imprimir mejor metrica al final del entrenamiento
		print('mejor dice score: ', best_dice)
		print('mejor iou score: ', best_iou)
		print('mejor pixel score: ', best_pixel)


		model.load_state_dict(best_model_wts)

		#extraer pesos del modelo
		model_weights = torch.save(model, 'model.pt')

		#visualizar grafica historica de perdidas
		plt.figure(figsize = (8,6))
		plt.plot(hist_epoch,hist_trn_loss,'r-',label = 'train')
		plt.plot(hist_epoch, hist_val_loss,'b-',label = 'validation')
		plt.title('histórico de pérdidas')
		plt.legend(['train', 'validation'])
		plt.xlabel('número de época')
		plt.savefig('./training/training_historics/LossHistorics.jpg')
		plt.clf()


		# visualizar grafica de metricas
		plt.figure(figsize = (8,6))
		plt.plot(hist_epoch,hist_dice_coeff,'r-',label = 'dice coeff')
		plt.plot(hist_epoch, hist_iou,'b-',label = 'IoU')
		plt.plot(hist_epoch, hist_pixel,'g-',label = 'pixel accuracy')
		plt.title('histórico de métricas')
		plt.legend(['dice_coeff', 'IoU', 'pixel accuracy'])
		plt.xlabel('número de época')
		plt.savefig('./training/training_historics/MetricsHistorics.jpg')

		return model, model_weights
