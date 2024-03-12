'''
Este modulo realiza un post procesamiento a las inferencias realizadas por el 
	modelo utilizando una operacion de apertura, eliminando artefactos ailados en
	la segmentación y afinandola.

'''


from cv2 import morphologyEx as M
import cv2
import numpy as np
import torch

class PostProcessing:

	def __init__(self, img):

		kernel = torch.ones(7, 7)

		self.post_img = M(np.float32(img), cv2.MORPH_OPEN, np.float32(kernel))

	def __getitem__(self):
		return self.post_img



