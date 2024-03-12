'''
Utilizar este módulo solamente en caso de que 
tengas dudas sobre el resultado del módulo de
"batch_data_processing"
'''

import matplotlib.pyplot as plt 
import numpy as np

from data_preparation import BatchDataProcessing

class Visualization:

	def __init__(self, examples):

		samples = enumerate(examples)
		batch_idx, (sample_data, sample_targets) = next(samples)

		fig = plt.figure(figsize = (10, 8))

		for i in range(4):
			plt.subplot(4, 2, i+(i+1))
			plt.tight_layout()
			plt.imshow(sample_data[i][0], cmap = 'gray', interpolation = 'none')
			plt.title("sample: {}".format(sample_data[i].max()))
			plt.xticks([])
			plt.yticks([])

			plt.subplot(4, 2, (2*(i+1)))
			plt.tight_layout()
			plt.imshow(sample_targets[i][0], cmap = 'gray', interpolation = 'none')
			plt.title("Ground truth: {}".format(sample_targets[i].max()))
			plt.xticks([])
			plt.yticks([])

		plt.show()

