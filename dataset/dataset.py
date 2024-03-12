'''

Este modulo solamente es para rescatar la direccion de los datasets

'''

import os

class Dataset:
	def files_path(data_name):
		abspath = os.path.abspath('dataset/volume_'+data_name+'.tif')
		# label_abspath = os.path.abspath('dataset/volume_label.tif')

		if data_name == 'real':
			filename = os.listdir('dataset/real_data')

			abspath = os.path.abspath('dataset/real_data/'+str(filename[0]))
			# print(abspath)

		# print(img_abspath,label_abspath)
		return abspath