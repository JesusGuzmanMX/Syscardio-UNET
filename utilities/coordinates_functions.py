from functools import reduce
import operator
import math
import numpy as np  

def outline(sample):

	topbottom = np.empty((1, 2*sample.shape[1]), dtype = np.uint16)
	topbottom[0,0:sample.shape[1]] = np.argmax(sample, axis = 0)
	topbottom[0, sample.shape[1]:] = (sample.shape[0] - 1) - np.argmax(np.flipud(sample), axis = 0)
	mask = np.tile(np.any(sample, axis = 0), (2,))
	xvalues = np.tile(np.arange(sample.shape[1]), (1,2))

	return np.vstack([topbottom, xvalues])[:, mask].T



def array2tuple_list(array):

	lista = []
	for i in range(array.shape[0]):
		tupla = (array[i, 1], array[i, 0])
		lista.append(tupla)

	return lista



def tuple_list2array(lista):

	array = np.zeros((len(lista) + 1, len(lista[0])))
	for i in range(array.shape[0] - 1):
		array[i, 0] = lista[i][0]
		array[i, 1] = lista[i][1]

	return array



def sort_coordinates(lista):
	coords = lista
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	lista_sorted = sorted(coords, key = lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)

	return lista_sorted



def coordinates(sample):
	lista = array2tuple_list(outline(sample))
	lista_sorted = sort_coordinates(lista)
	n = len(lista_sorted)
	sample_coords = tuple_list2array(lista_sorted)
	sample_coords[n, :] = sample_coords[0, :]

	return sample_coords



def coordinates_dict(sample ,k, cd):

	# keys del diccionario
	heart_keys = ["LV", "LA", "RV", "RA"]

	if k == 0:
		#crear diccionario
		coordenadas = {}
	else:
		coordenadas = cd

	sample_coords = coordinates(sample)
	sc = sample_coords.tolist()

	new_key = heart_keys[k]
	new_value = sc 
	coordenadas[new_key] = new_value

	return coordenadas