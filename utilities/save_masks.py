import matplotlib.pyplot as plt

class SaveMasks:

	def __init__(self, sample, c):
		plt.imsave('./results/masks/' + str(c) + '.tiff', sample, cmap = 'gray')

