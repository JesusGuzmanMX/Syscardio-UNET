import torch

class SplitClass:

	def split_classes(preds, target):

		# tensor de tamaño [4 x 128 x 128]
		pred = torch.zeros(4, 128, 128)
		true = torch.zeros(4, 128, 128)

		# tensor con clases [0, 1, 2, 3, 4] (BG, VI, AI, VD, AD)
		classes = torch.unique(target.int())


		# ciclo for para cada cavidad [VI, AI, VD, AD]
		for i in range(1, len(classes)):
			pred[i-1, :, :] = (preds == classes[i]).float()
			true[i-1, :, :] = (target == classes[i].float())

		return pred, true