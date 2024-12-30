import numpy as np
import h5py
import sys
import yaml


cache = {}

#data loader
def load_data(name):
	with h5py.File(f"/Users/micaelaroth/Downloads/local_copy/scripts/relevant/vesicle_big_{name}_30-8-8.h5", "r") as f:
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

	v = np.zeros(vesicles.shape, dtype=vesicles.dtype) #new file

	#first relabel for vesicles 1, 2
	v[vesicles==1] = d[1]
	v[vesicles==2] = d[2]

	for label, new_label in d.items():
		#avoid relabeling vesicles 0, 1, 2 again
		if (label not in (0,1,2)):
			v[(vesicles==label) & (v!=new_label)] = new_label
	return v


#main
if __name__ == "__main__":
	name = "LUX2"

	load_data(name)
	print(f"done loading data for {name}")

	#print num of components in vesicles
	data = cache["vesicles"].flatten()
	unique_labels = np.unique(data)
	print("num vesicles: ", len(unique_labels)-1) #remove bg label

	dictionary = read_txt_to_dict(f"/Users/micaelaroth/Downloads/local_copy/{name}_types.txt")
	print("original num of keys: ", len(dictionary.keys()))

	#remove everything with value 0
	remove = [k for k, v in dictionary.items() if v==0]
	for k in remove:
		del dictionary[k]

	print("new num of keys: ", len(dictionary.keys()))

	color_coded = generate_color_coded(cache["vesicles"], dictionary)
	print("done generating color coded file")


	with h5py.File(f"/Users/micaelaroth/Downloads/local_copy/{name}_color_coded.h5", "w") as f:
		f.create_dataset("main", shape=color_coded.shape, data=color_coded)
		print(f"saved as {name}_color_coded.h5")




