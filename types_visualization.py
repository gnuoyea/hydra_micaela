import numpy as np
import h5py
import sys
import yaml
import argparse


cache = {}
D0 = '/data/projects/weilab/dataset/hydra/results/'
D4 = '/data/rothmr/hydra/types_lists/' #full types lists
D5 = '/data/rothmr/hydra/color_coded/' #output dir for files


#data loader
def load_data(name):
	with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", "r") as f:
		cache["vesicles"] = f["main"][:]


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


#generate the file with different segids for each vesicle type
def generate_color_coded(vesicles, d):
	#print num of unique labels in vesicles
	unique_labels = np.unique(vesicles)
	num_labels = len(unique_labels) - 1 #minus the bg label

	print("checkpoint 0")

	v = np.zeros(vesicles.shape, dtype=vesicles.dtype) #new file

	#first relabel for vesicles 1, 2
	v[vesicles==1] = d[1]
	v[vesicles==2] = d[2]

	print("checkpoint 1")

	for label, new_label in d.items():
		print("label: ", label, "new label: ", new_label)
		#avoid relabeling vesicles 0, 1, 2 again
		if (label not in (0,1,2)):
			#v[(vesicles==label) & (v!=new_label)] = new_label
			v[vesicles==label] = new_label
	print("loop done")
	return v


#doesn't ever load full data
def load_chunks(name, chunk_num, num_chunks):
	print(f"begin loading chunk #{chunk_num+1}") #bc zero indexing
	chunk_length = 0 #initialize for scope

	path = f'{D0}vesicle_big_{name}_30-8-8.h5' #use name param
	with h5py.File(path, 'r') as f:
		shape = f["main"].shape
		dtype = f["main"].dtype

		#calculate chunk_length (last chunk might be this plus some remainder if this doesn't divide evenly)
		chunk_length = (shape[1])//num_chunks #integer division, dividing up y axis length

		if(chunk_num!=num_chunks-1):
			cache["chunk"] = f["main"][:, chunk_num*chunk_length:(chunk_num+1)*chunk_length, :]

		else: #case of the last chunk
			cache["chunk"] = f["main"][:, chunk_num*chunk_length:, :] #go to end of the file - last chunk includes any leftover stuff

		print(f"done loading chunk #{chunk_num+1} of {num_chunks}") #bc zero indexing

	return chunk_length


#main
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='neuron name')
	args = parser.parse_args()
	name = args.name
	#name = "KR5"

	chunking = True
	num_chunks = 8


	if(chunking):
		current_chunk = 0
		path = f"{D0}vesicle_big_{name}_30-8-8.h5"
		with h5py.File(path, 'r') as f:
			output = np.zeros(shape=f["main"].shape, dtype=f["main"].dtype) #initialize the full output thing

		while(current_chunk!=num_chunks): #runs [num_chunk] times
			chunk_length = load_chunks(name, current_chunk, num_chunks)
			chunk = cache["chunk"]

			#generate color coded for the  chunk
			data = chunk.flatten()
			unique_labels = np.unique(data)
			dictionary = read_txt_to_dict(f"{D4}{name}_types.txt")
			remove = [k for k, v in dictionary.items() if v==0]
			for k in remove:
				del dictionary[k]
			print("checkpoint")

			color_coded_chunk = generate_color_coded(chunk, dictionary) #not padded
			print(f"done generating color coded chunk #{current_chunk+1}")

			#find coords of current chunk
			y_start = current_chunk * chunk_length

			#cases for last chunk or not + insert chunk into place
			if(current_chunk!=num_chunks-1): #not last chunk
				output[:, y_start:(current_chunk+1)*chunk_length, :] = color_coded_chunk
			else: #last chunk
				output[:, y_start:, :] = color_coded_chunk

			current_chunk+=1

		#save the output file to an h5
		print("done generating color coded file")

		with h5py.File(f"{D5}{name}_color_coded.h5", "w") as f:
			f.create_dataset("main", shape=output.shape, data=output)
			print(f"saved as {name}_color_coded.h5")


	else:
		load_data(name)
		#print num of components in vesicles
		data = cache["vesicles"].flatten()
		unique_labels = np.unique(data)
		print("num vesicles: ", len(unique_labels)-1) #remove bg label

		dictionary = read_txt_to_dict(f"{D4}{name}_types.txt")
		print("original num of keys: ", len(dictionary.keys()))

		#remove everything with value 0
		remove = [k for k, v in dictionary.items() if v==0]
		for k in remove:
			del dictionary[k]

		print("new num of keys: ", len(dictionary.keys()))

		color_coded = generate_color_coded(cache["vesicles"], dictionary)
		print("done generating color coded file")

		with h5py.File(f"{D5}{name}_color_coded.h5", "w") as f:
			f.create_dataset("main", shape=color_coded.shape, data=color_coded)
			print(f"saved as {name}_color_coded.h5")




