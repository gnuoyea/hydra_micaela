import numpy as np
import h5py
import yaml

#use normalized mappings generated in density_new.py
#generate color coded file color_test.h5


D0 = '/data/projects/weilab/dataset/hydra/results/'
cache = {}

def to_dict(file_path):
	data_dict = {}
	with open(file_path, 'r') as file:
		for line in file:
			line = line.strip()
			if line:
				key, value = line.split(":")
				data_dict[int(key)] = float(value)
	return data_dict


#load data
def load_data():
	#just load kr6 LV for test
	with h5py.File(f"{D0}vesicle_big_KR6_30-8-8.h5", 'r') as f:
		cache["lv"] = np.array(f["main"][:][:, ::2, ::2]) #30-16-16 for vis
	print("data loaded for kr6")


#based on percentiles extracted from the actual data
def convert(density, percentiles):
	#yellow - assign 1
	if(density>0 and density<=percentiles[0]):
		label = 1
	#orange - assign 2
	elif(density>percentiles[0] and density<=percentiles[1]):
		label = 2
	#red - assign 3
	elif(density>percentiles[1]):
		label = 3
	else:
		label = 0
	return label


'''
#optional
def plot(d):
	data = np.ndarray(d.values())
	plt.hist(data, bins=10, edgecolor='black', alpha=0.7)
	plt.show()
'''


def generate(vesicles, d):
	#squish dict into a Gaussian distribution
	percentiles = np.percentile(list(d.values()), [33, 66])
	print("percentiles: ", percentiles)

	unique_labels = np.unique(vesicles)
	num_labels = len(unique_labels) - 1 #minus the bg label
	print("checkpoint 0")

	v = np.zeros(vesicles.shape, dtype=vesicles.dtype) #new file

	#first relabel for vesicles 1, 2, 3
	if np.any(vesicles == 1):
		v[vesicles==1] = convert(d[1], percentiles)
	if np.any(vesicles == 2):
		v[vesicles==2] = convert(d[2], percentiles)
	if np.any(vesicles == 3):
		v[vesicles==3] = convert(d[3], percentiles)

	print("checkpoint 1")

	for label, new_label in d.items():
		print("label: ", label, "new label: ", convert(new_label, percentiles))
		#avoid relabeling vesicles 0, 1, 2, 3 again
		if (label not in (0,1,2,3)):
			v[vesicles==label] = convert(new_label, percentiles)
	print("loop done")
	
	#save "v" as a new file
	with h5py.File(f"color_test.h5", "w") as f:
		f.create_dataset("main", shape=v.shape, data=v)
		print(f"saved file as color_test.h5")


if __name__ == "__main__":
	path = "KR6_density_mapping.txt"
	d = to_dict(path)
	print("dict initialized")
	load_data()
	generate(cache["lv"], d) #saves file








