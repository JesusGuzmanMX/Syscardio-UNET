from utilities import SplitClass

import torch



def ind_iou(preds, target):
	inter = (preds * target).sum()
	union = (preds + target).sum() - inter
	return inter / (union + 1e-8)

def int_union(preds, target):
	#dividir clases
	pred, true = SplitClass.split_classes(preds, target)

	# tensor para almacenar los iou
	iou_class = torch.zeros(4)

	#ciclo for para calcular el iou para cada clase
	for i in range(pred.shape[0]):
		iou_class[i] = ind_iou(pred[i], true[i])

	# media de iou de todas las clases
	iou_total = torch.mean(iou_class)

	return iou_total

def iou_batch(model_preds, model_targets):

	#revisar si esta en gpu o no
	if model_preds.is_cuda:
		s = torch.FloatTensor(1).cuda().zero_()
	else:
		s = torch.FloatTensor(1).zero_()

	# i es el indice y c el array
	for i, c in enumerate(zip(model_preds, model_targets)):
		s = s + int_union(torch.argmax(c[0], dim = 0), c[1])

	return s/(i+1)