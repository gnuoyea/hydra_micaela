import h5py
import numpy as np
import scipy.stats as stats

cache = {}

D0 = '/data/projects/weilab/dataset/hydra/results/'
D1 = '/data/rothmr/hydra/stitched/'
D_volumes = '/data/rothmr/hydra/volumes/' #for raw list of volumes for each neuron

#dictionaries for SV type classifications will be saved in cache
names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", "RGC2", "KM4", "SHL17",
				"NET12", "NET10", "NET11", "PN7", "SHL18", "SHL24", "SHL28", "RGC7"]

for name in names_20:
	dictionary = dict()
	cache[f"{name}_SV_dict"] = dictionary


def load_data(name, lv=False, sv=False):
	print(f"begin loading data for {name}")
	if(lv):
		with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f: #high res large vesicles data for [name]
			cache["lv"] = f["main"][:]
		print("done loading LV")

	if(sv):
		with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f: #high res small vesicles data for [name]
			cache["sv"] = f["main"][:]
		print("done loading SV")

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


def calculate_stats(name, lv=False, sv=False):
	dictionary = read_txt_to_dict(f"types_lists/{name}_types.txt")
	res = [30,8,8]
	lv_volumes = []
	sv_volumes = []

	#load the data into the cache for this neuron
	load_data(name, lv, sv)

	#calculate distribution of vesicle volumes for lv and sv separately

	if(lv):
		#find list of unique labels
		unique_labels = np.unique(cache["lv"])
		print("LV num of unique labels including 0: ", len(unique_labels))
		voxel_volume = res[0]*res[1]*res[2]

		for label in unique_labels:
			if label!=0:
				mask = (cache["lv"] == label) #binary mask
				volume_voxels = np.sum(mask)
				volume = volume_voxels * voxel_volume
				#print(volume)
				lv_volumes.append(volume)
		lv_volumes = np.array(lv_volumes)
		mean = np.mean(lv_volumes)
		sem = stats.sem(lv_volumes)
		n = len(lv_volumes)
		print(f"Total LV mean: {mean}, LV sem: {sem}, LV n: {n}")

		#save all VOLUMES to a file
		with open(f"{D_volumes}{name}_LV_volumes.txt", "w") as f:
			for v in lv_volumes:
				f.write(f"{v}\n")

		# cv = 1
		cv_volumes = []
		for label in unique_labels:
			if label!=0:
				vesicle_subtype = dictionary[label]
				if vesicle_subtype==1:
					mask = (cache["lv"] == label) #binary mask
					volume_voxels = np.sum(mask)
					volume = volume_voxels * voxel_volume
					cv_volumes.append(volume)
		cv_volumes = np.array(cv_volumes)
		mean = np.mean(cv_volumes)
		sem = stats.sem(cv_volumes)
		n = len(cv_volumes)
		print(f"CV mean: {mean}, CV sem: {sem}, CV n: {n}")

		# dv = 2
		dv_volumes = []
		for label in unique_labels:
			if label!=0:
				vesicle_subtype = dictionary[label]
				if vesicle_subtype==2:
					mask = (cache["lv"] == label) #binary mask
					volume_voxels = np.sum(mask)
					volume = volume_voxels * voxel_volume
					dv_volumes.append(volume)
		dv_volumes = np.array(dv_volumes)
		mean = np.mean(dv_volumes)
		sem = stats.sem(dv_volumes)
		n = len(dv_volumes)
		print(f"DV mean: {mean}, DV sem: {sem}, DV n: {n}")

		# dvh = 3
		dvh_volumes = []
		for label in unique_labels:
			if label!=0:
				vesicle_subtype = dictionary[label]
				if vesicle_subtype==3:
					mask = (cache["lv"] == label) #binary mask
					volume_voxels = np.sum(mask)
					volume = volume_voxels * voxel_volume
					dvh_volumes.append(volume)
		dvh_volumes = np.array(dvh_volumes)
		mean = np.mean(dvh_volumes)
		sem = stats.sem(dvh_volumes)
		n = len(dvh_volumes)
		print(f"DVH mean: {mean}, DVH sem: {sem}, DVH n: {n}")

		
	if(sv):
		#fill in dicts for all neurons
		with np.load("sv_types/SV_types_new.npz") as data:
			current = 0
			while (current<len(data["ids"])):
				name = data["ids"][current][0]
				vesicle_id = int(data["ids"][current][1]) #vesicle ID in the [name] neuron
				label = data["labels"][current] #type from the embeddings, float type

				#SDV
				if(label==0):
					vesicle_type = 4
				#SCV
				if(label==1):
					vesicle_type = 5

				if(label!=0 and label!=1):
					print("label error")

				cache[f"{name}_SV_dict"] = {**cache[f"{name}_SV_dict"], vesicle_id:vesicle_type} #update dict for the [name] neuron
				current+=1;

			print("initialized all types mapping dictionaries")


		#find list of unique labels
		unique_labels = np.unique(cache["sv"])
		print("SV num of unique labels including 0: ", len(unique_labels))
		voxel_volume = res[0]*res[1]*res[2]

		for label in unique_labels:
			if label!=0:
				mask = (cache["sv"] == label) #binary mask
				volume_voxels = np.sum(mask)
				volume = volume_voxels * voxel_volume
				#print(volume)
				sv_volumes.append(volume)

		sv_volumes = np.array(sv_volumes)
		mean = np.mean(sv_volumes)
		sem = stats.sem(sv_volumes)
		n = len(sv_volumes)
		print(f"Total SV mean: {mean}, SV sem: {sem}, SV n: {n}")

		#save all VOLUMES to a file
		with open(f"{D_volumes}{name}_SV_volumes.txt", "w") as f:
			for v in sv_volumes:
				f.write(f"{v}\n")

		#now split into types
		sdv_volumes = []
		for label in unique_labels:
			if label!=0:
				vesicle_subtype = cache[f"{name}_SV_dict"][label]
				if vesicle_subtype==4:
					mask = (cache["sv"] == label) #binary mask
					volume_voxels = np.sum(mask)
					volume = volume_voxels * voxel_volume
					sdv_volumes.append(volume)
		sdv_volumes = np.array(sdv_volumes)
		mean = np.mean(sdv_volumes)
		sem = stats.sem(sdv_volumes)
		n = len(sdv_volumes)
		print(f"SDV mean: {mean}, SDV sem: {sem}, SDV n: {n}")

		scv_volumes = []
		for label in unique_labels:
			if label!=0:
				vesicle_subtype = cache[f"{name}_SV_dict"][label]
				if vesicle_subtype==5:
					mask = (cache["sv"] == label) #binary mask
					volume_voxels = np.sum(mask)
					volume = volume_voxels * voxel_volume
					scv_volumes.append(volume)
		scv_volumes = np.array(scv_volumes)
		mean = np.mean(scv_volumes)
		sem = stats.sem(scv_volumes)
		n = len(scv_volumes)
		print(f"SCV mean: {mean}, SCV sem: {sem}, SCV n: {n}")

		


if __name__ == "__main__":
	to_generate = names_20
	for name in to_generate:
		calculate_stats(name, lv=True, sv=True) #for sv














