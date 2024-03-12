import json

import utilities.coordinates_functions as CoordinatesFunction

class SaveCoordinates:

	def __init__(self, sample, c):

		with open('./results/json_files/' + str(c) + '.json', "w") as outfile:
			json.dump(sample, outfile)
