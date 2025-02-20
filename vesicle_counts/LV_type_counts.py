import numpy as np
import h5py
import re
import argparse

def read_txt_to_dict(file_path):
	result_dict = {}
	with open(file_path, 'r') as file:
		for line in file:
			line = line.strip()
			pairs = line.strip('()').split('),(')

			for pair in pairs:
				key, value = pair.split(':')
				result_dict[int(key)] = int(value)

	return result_dict

def read_txt_to_list(file_path):
	result_list = []
	with open(file_path, 'r') as file:
		for line in file:
			line = re.sub(r'[\[\]{}]','',line)
			result_list.extend(map(int, line.strip().split()))

	return result_list

if __name__ == "__main__":
	#neurons_list = ['KR4']

	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='neuron name')
	args = parser.parse_args()
	name = args.name

	neurons_list = [name]


	for name in neurons_list:
		print(f"----{name}----")
		dictionary = read_txt_to_dict(f"types_lists/{name}_types.txt")
		lst = read_txt_to_list(f"lists/within_{name}.txt")
		print("length of dict: ", len(dictionary.keys()))
		print("length of list: ", len(lst))

		#total by type count stuff
		print("---total counts---")
		total_num_unrecognized=0
		total_num_CV=0
		total_num_DV=0
		total_num_DVH=0

		for vesicle in dictionary.keys():
			if(dictionary[vesicle]==0):
				total_num_unrecognized+=1
			if(dictionary[vesicle]==1):
				total_num_CV+=1
			if(dictionary[vesicle]==2):
				total_num_DV+=1
			if(dictionary[vesicle]==3):
				total_num_DVH+=1

		print(f"CV: {total_num_CV}, DV: {total_num_DV}, DVH: {total_num_DVH}, unrecognized: {total_num_unrecognized}")
		print("total length check: ", total_num_unrecognized+total_num_CV+total_num_DV+total_num_DVH==len(dictionary.keys()))


		print("---near neuron counts---")
		num_unrecognized=0
		num_CV=0
		num_DV=0
		num_DVH=0

		for vesicle in lst:
			if vesicle in dictionary.keys():
				if(dictionary[vesicle]==0):
					num_unrecognized+=1
				if(dictionary[vesicle]==1):
					num_CV+=1
				if(dictionary[vesicle]==2):
					num_DV+=1
				if(dictionary[vesicle]==3):
					num_DVH+=1
			else:
				print(f"error in mapping at key {vesicle}")

		print(f"CV: {num_CV}, DV: {num_DV}, DVH: {num_DVH}, unrecognized: {num_unrecognized}")
		print("total length check: ", num_unrecognized+num_CV+num_DV+num_DVH==len(lst))








