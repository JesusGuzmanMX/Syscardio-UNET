from utilities import SplitClass

import torch


def ind_pixel(preds, target):
	return (preds * target).sum() /(target.sum() + 1e-8)

def pixel_accuracy(preds, target):
	# dividir clases
	pred, true = SplitClass.split_classes(preds, target)

	# tensor para almacenar los PA
	pixel_class = torch.zeros(4)

	# ciclo for para calcular el PA para cada clase
	for i in range(pred.shape[0]):
		pixel_class[i] = ind_pixel(pred[i], true[i])

	# media de PA de todas las clases
	pixel_total = torch.mean(pixel_class)

	return pixel_total


def pixel_batch(model_preds, model_targets):

	# revisar si está en gpu o no
	if model_preds.is_cuda:
		s = torch.FloatTensor(1).cuda().zero_()
	else:
		s = torch.FloatTensor(1).zero_()

	# i es el indice y c el array
	for i, c in enumerate(zip(model_preds, model_targets)):
		s = s + pixel_accuracy(torch.argmax(c[0], dim = 0), c[1])

	return s/(i+1)
