from utilities import SplitClass

import torch

def ind_dice(preds, target):
	return (2 * (preds * target).sum()) / (
				(preds + target).sum() + 1e-8)

def dice_coef(preds, target):

	#dividir clases
	pred, true = SplitClass.split_classes(preds, target)

	#tensor para almacenar los dice coeff
	dice_class = torch.zeros(4)

	# ciclo for para calcular el dice coeff para cada clase
	for i in range(pred.shape[0]):
		dice_class[i] = ind_dice(pred[i], true[i])


	# media de dice coeff de todas las clases
	dice_total = torch.mean(dice_class)

	return dice_total


def dice_batch(model_preds, model_targets):

	#revisar si está en gpu o no

	if model_preds.is_cuda:
		s = torch.FloatTensor(1).cuda().zero_()
	else:
		s = torch.FloatTensor(1).zero_()

	# i es indice y c array
	for i, c in enumerate(zip(model_preds, model_targets)):
		s = s + dice_coef(torch.argmax(c[0], dim = 0), c[1])

	return s/(i+1)
