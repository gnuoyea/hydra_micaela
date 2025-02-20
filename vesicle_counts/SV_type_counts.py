import numpy as np
from numpy import load
import h5py
import re

#different script from lv type counts since sv data/mappings are in different format

cache = {}

#initialize dicts in cache
names_13 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", "RGC2", "KM4", "SHL17", "NET12"]

names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", "RGC2", "KM4", "SHL17",
				"NET12", "NET10", "NET11", "PN7", "SHL18", "SHL24", "SHL28", "RGC7"]

names_7 = ["NET10", "NET11", "PN7", "SHL18", "SHL24", "SHL28", "RGC7"]

for name in names_20:
	dictionary = dict()
	cache[f"{name}"] = dictionary


def read_txt_to_list(file_path):
	result_list = []
	with open(file_path, 'r') as file:
		for line in file:
			line = re.sub(r'[\[\]{}]','',line)
			result_list.extend(map(int, line.strip().split()))

	return result_list


if __name__ == "__main__":
	
	#fill in dicts for all neurons
	with np.load("sv_types/SV_types.npz") as data:
		'''
		#check to make sure same length
		print("length of data[ids]: ", len(data["ids"]))
		print("length of data[embeddings]: ", len(data["embeddings"]))
		'''

		current = 0
		while (current<len(data["ids"])):
			name = data["ids"][current][0]
			vesicle_id = int(data["ids"][current][1]) #vesicle ID in the [name] neuron
			embedding = data["embeddings"][current] #type from the embeddings, float type

			#SDV
			if(embedding>0):
				vesicle_type = 4
			#SCV
			if(embedding<0):
				vesicle_type = 5


			cache[name] = {**cache[name], vesicle_id:vesicle_type} #update dict for the [name] neuron
			current+=1;

		print("initialized all types mapping dictionaries")


	neurons_list = ["SHL28"]

	#print out counts for each neuron dict
	for name in neurons_list:
		print(f"----{name}----")
		dictionary = cache[name] #format is vid:type for this neuron 
		lst = read_txt_to_list(f"sv_nn/sv_nn_{name}.txt") #near neuron IDs list

		print("total num vesicles: ", len(dictionary.keys()))
		print("total NN vesicles: ", len(lst)) #near neuron

		#total by type count stuff
		print("---total counts---")
		total_num_SDV=0 #positive embedding, black
		total_num_SCV=0 #negative embedding, white

		for vesicle in dictionary.keys():
			if(dictionary[vesicle]==4):
				total_num_SDV+=1
			if(dictionary[vesicle]==5):
				total_num_SCV+=1

		print(f"SCV: {total_num_SCV}, SDV: {total_num_SDV}")
		print("total length check: ", total_num_SDV+total_num_SCV==len(dictionary.keys()))


		print("---near neuron counts---")
		nn_SDV = 0
		nn_SCV = 0

		for vesicle in lst: #the near neuron IDs list
			if vesicle in dictionary.keys():
				if(dictionary[vesicle]==4):
					nn_SDV+=1
				if(dictionary[vesicle]==5):
					nn_SCV+=1

			else:
				print(f"error in mapping at key {vesicle}")

		print(f"nn SCV: {nn_SCV}, nn SDV: {nn_SDV}")
		print("total length check: ", nn_SDV+nn_SCV==len(lst))








