import edt
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
import os
import yaml

cache = {}

D0 = '/data/projects/weilab/dataset/hydra/results/'
D1 = '/data/rothmr/hydra/stitched/'
D2 = '/data/rothmr/hydra/lists/'

#all files should be in the shape of the neuron mask file for the neuron name eg. "KR5"
#use for non chunking pipeline
def load_data(name):
	print("begin loading data")
	with h5py.File(f"{D1}neuron_{name}_box_30-32-32.h5", 'r') as f: #stitched mask, contains everything w segids
		cache["box"] = f["main"][:]
	print("done loading neuron box")

	with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f: #high res large vesicles data for [name]
		cache["lv"] = f["main"][:]
	print("done loading LV")

	with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f: #high res small vesicles data for [name]
		cache["sv"] = f["main"][:]
	print("done loading SV")

#use for chunking pipeline
def load_neurons(name):
	print("begin loading data")
	with h5py.File(f"{D1}neuron_{name}_box_30-32-32.h5", 'r') as f: #stitched mask, contains everything w segids
		cache["box"] = f["main"][:]
	print("done loading neuron box")

#always splitting along the y axis; index starting at 'chunk 0'
#chunks are not padded -> coords are adjusted in the calculate_vesicles_within function
#returns the chunk_length variable for offset purposes
def load_lv_chunks(chunk_num, num_chunks):
	print(f"begin loading LV chunk #{chunk_num+1}") #bc zero indexing
	chunk_length = 0 #initialize for scope
	with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f: #high res large vesicles data for [name]
		shape = f["main"].shape
		dtype = f["main"].dtype

		#calculate chunk_length (last chunk might be this plus some remainder if this doesn't divide evenly)
		chunk_length = (shape[1])//num_chunks #integer division, dividing up y axis length

		if(chunk_num!=num_chunks-1):
			cache["lv_chunk"] = f["main"][:, chunk_num*chunk_length:(chunk_num+1)*chunk_length, :]

		else: #case of the last chunk
			cache["lv_chunk"] = f["main"][:, chunk_num*chunk_length:, :] #go to end of the file - last chunk includes any leftover stuff

	print(f"done loading LV chunk #{chunk_num+1}") #bc zero indexing

	return chunk_length


#same logic as load_lv_chunks
def load_sv_chunks(chunk_num, num_chunks):
	print(f"begin loading SV chunk #{chunk_num+1}") #bc zero indexing
	chunk_length = 0 #initialize for scope
	with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f: #high res large vesicles data for [name]
		shape = f["main"].shape
		dtype = f["main"].dtype

		#calculate chunk_length (last chunk might be this plus some remainder if this doesn't divide evenly)
		chunk_length = (shape[1])//num_chunks #integer division, dividing up y axis length

		if(chunk_num!=num_chunks-1):
			cache["sv_chunk"] = f["main"][:, chunk_num*chunk_length:(chunk_num+1)*chunk_length, :]

		else: #case of the last chunk
			cache["sv_chunk"] = f["main"][:, chunk_num*chunk_length:, :] #go to end of the file - last chunk includes any leftover stuff

	print(f"done loading SV chunk #{chunk_num+1}") #bc zero indexing

	return chunk_length


def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data


#calculate num of vesicles within any given mask; takes in mask and corresponding vesicles file
#vesicles file is in higher res
#if chunking - accounts for potential overlap vesicles across chunks
def calculate_vesicles_within(mask, vesicles=None, save_list=False, name=None, chunking=False, num_chunks=0, sv=False, lv=False):
	bool_mask = (mask>=1)
	bool_mask = zoom(bool_mask, (1, 4, 4), order=0) #change into ves file res
	print("done changing mask file resolution")

	if(chunking): #find lists of seg ids within the mask, for each chunk. merge uniquely into one list, then return final list length
		if(lv):
			within_ids_set = set() #within the mask
			total_ids_set = set() #within the full vesicles file

			#loop thru each chunk and extend list every time
			current_chunk = 0
			while(current_chunk!=num_chunks): #this loop runs [num_chunks] times
				chunk_length = load_lv_chunks(current_chunk, num_chunks)
				y_start = current_chunk * chunk_length

				#cases for last chunk or not
				if(current_chunk!=num_chunks-1): #not last chunk
					cache["mask_chunk"] = bool_mask[:, y_start:(current_chunk+1)*chunk_length, :]
				else: #last chunk
					cache["mask_chunk"] = bool_mask[:, y_start:, :]

				labeled_vesicles = cache["lv_chunk"]
				mask_chunk = cache["mask_chunk"]
				print("checkpoint 1")

				vesicle_coords = np.column_stack(np.nonzero(labeled_vesicles))
				print("checkpoint 2")

				unique_labels = set(labeled_vesicles.ravel())
				total_ids_set.update(unique_labels) #updating our set

				mask_values = mask_chunk[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]]
				print("checkpoint 3")

				vesicles_within_mask = set(labeled_vesicles[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]][mask_values])
				within_ids_set.update(vesicles_within_mask)

				current_chunk += 1

			print("total num LV: ", len(total_ids_set)-1) #minus bg label 0
			
			#only for LV
			if(save_list):
				#export list as a txt file
				path = f'{D2}within_{name}.txt'
				with open(path, "w") as f:
					f.write(str(list(within_ids_set)))
				print("list for types exported")

			return len(within_ids_set) #num of unique ids within the mask

		#same logic as for lv
		if(sv):
			within_ids_set = set() #within the mask
			total_ids_set = set() #within the full vesicles file

			#loop thru each chunk and extend list every time
			current_chunk = 0
			while(current_chunk!=num_chunks): #this loop runs [num_chunks] times
				chunk_length = load_sv_chunks(current_chunk, num_chunks)
				y_start = current_chunk * chunk_length

				#cases for last chunk or not
				if(current_chunk!=num_chunks-1): #not last chunk
					cache["mask_chunk"] = bool_mask[:, y_start:(current_chunk+1)*chunk_length, :]
				else: #last chunk
					cache["mask_chunk"] = bool_mask[:, y_start:, :]

				labeled_vesicles = cache["sv_chunk"]
				mask_chunk = cache["mask_chunk"]
				print("checkpoint 1")

				vesicle_coords = np.column_stack(np.nonzero(labeled_vesicles))				
				print("checkpoint 2")

				unique_labels = set(labeled_vesicles.ravel())
				total_ids_set.update(unique_labels) #updating our set

				mask_values = mask_chunk[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]]
				print("checkpoint 3")

				vesicles_within_mask = set(labeled_vesicles[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]][mask_values])
				within_ids_set.update(vesicles_within_mask)

				current_chunk += 1

			print("total num SV: ", len(total_ids_set)-1) #minus bg label 0

			return len(within_ids_set)

	else: #no chunking
		num_vesicles_within = 0

		labeled_vesicles = vesicles
		vesicle_coords = np.column_stack(np.nonzero(labeled_vesicles))

		unique_labels = np.unique(labeled_vesicles)
		num_labels = len(unique_labels) - 1 #minus bg label 0
		print("total num of vesicles: ", num_labels)

		vesicles_within_mask = np.zeros(num_labels, dtype=bool)
		mask_values = bool_mask[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]]
		vesicles_within_mask = np.unique(labeled_vesicles[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]][mask_values])
		num_vesicles_within = len(vesicles_within_mask)

		if(save_list):
			#export vesicles_within_mask as a txt file
			path = f'{D2}within_{name}.txt'
			with open(path, "w") as f:
				f.write(str(vesicles_within_mask))
			print("list for types exported")

		return num_vesicles_within


#use if neurons and vesicles files have different res (testing phase)
def convert_coords(original_coords, original_shape, original_res, target_shape, target_res):
    relative_coord = np.array(original_coords) / np.array(original_shape)
    target_coord = np.round(relative_coord * np.array(target_shape)).astype(int)
    return target_coord

def expand_mask(original_mask, threshold_nm, res): #threshold in physical units, res in xyz
	expanded_mask = np.copy(original_mask)

	print("begin distance transform for expansion")
	dt = edt.edt(1 - original_mask, anisotropy=(res[2],res[1],res[0])) #needs to be in xyz by default for edt
	print("end distance transform for expansion")
	doubled_perimeter = dt <= threshold_nm #all in nm
	expanded_mask[doubled_perimeter] = 1

	expanded_mask_binary = (expanded_mask>=1).astype(int)
	return expanded_mask_binary


#for usage for a singular mask, docked vesicle counts
def erode_mask(original_mask, threshold_nm, res):
	eroded_mask = np.copy(original_mask)

	dt = edt.edt(1 - original_mask, anisotropy=(res[2],res[1],res[0])) #needs to be in xyz by default for edt
	perimeter = dt <= threshold_nm #all in nm; double perim
	eroded_mask[perimeter] = 0

	eroded_mask_binary = (eroded_mask>=1).astype(int)
	return eroded_mask_binary


def docked_vesicles(original_mask, vesicles, threshold_nm, res): #use 250nm
	eroded_mask = erode_mask(original_mask, threshold_nm, res)
	return calculate_vesicles_within(eroded_mask, vesicles), calculate_volume_nm(eroded_mask, res)


#for visualization uses only
def extract_perimeter(expanded_mask, original_mask):
	#expanded mask should already be binary
	#change original mask to binary
	original_mask_binary = (original_mask>=1).astype(int)
	original_mask_inverted = ~original_mask_binary

	#return everything in expanded but NOT in original
	perimeter_only_mask = np.copy(expanded_mask)
	perimeter_only_mask[perimeter_only_mask & original_mask_inverted] = 0

	return perimeter_only_mask


#takes in a list of masks, outputs a binary mask
def mask_intersection(mask_list): #assume same size files, assume list has size at least 1
	intersection = np.bitwise_and.reduce(mask_list)
	return intersection


#calculate the volume given a boolean mask & convert to nm^3 units
def calculate_volume_nm(mask, res):
	binary_mask = (mask>=1).astype(int)
	volume_voxels = np.sum(binary_mask)
	volume_nm = volume_voxels * (res[0]*res[1]*res[2])
	return volume_nm


def neuron_name_to_id(name):
	if isinstance(name, str):
		name = [name]
	return [neuron_dict[x] for x in name]  

if __name__ == "__main__":
	neuron_dict = read_yml('/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt')
	res1 = (30,8,8)
	res2 = (30,32,32)

	chunking = True #use chunking only if memory issues

	to_calculate = ['SHL17']

	for name in to_calculate:
		print(f"-----------{name}-----------")
		nid = neuron_name_to_id(name)
		threshold = 1000 #nm
		#docked_threshold = 250

		if(chunking):
			#only load neurons, load vesicles later in chunks
			load_neurons(name)

		else:
			#load everything
			load_data(name)

		print("done loading data")

		neurons = cache["box"]
		#unique_ids = np.unique(neurons)

		neuron_only = np.zeros(neurons.shape, dtype=neurons.dtype)
		neuron_only[neurons==nid] = 1 #binary mask
		print(f"Volume (in nm^3) of {name} only: ", calculate_volume_nm(neuron_only, res2))

		other_neurons = np.zeros(neurons.shape, dtype = neurons.dtype)
		other_neurons[(neurons!=nid) & (neurons!=0)] = 1 #binary mask
		expanded_others = expand_mask(other_neurons, threshold, res2)
		print("mask expanding done")
		intersections = mask_intersection((neuron_only, expanded_others))
		print("mask intersection done")
		print(f"Volume (in nm^3) of intersections for {name} and {threshold}nm: ", calculate_volume_nm(intersections, res2))
		print("\ndone initializing everything\n")


		if(chunking):
			#implement chunking
			num_chunks = 4
			print(f"calculate using chunking with {num_chunks} chunks")
			
			print("---LV---")
			print(f"LV within neuron {name} & within {threshold}nm of another neuron: ", 
				calculate_vesicles_within(intersections, save_list=False, name=name, chunking=True, num_chunks=num_chunks, lv=True))
			
			print("---SV---")
			print(f"SV within neuron {name} & within {threshold}nm of another neuron: ", 
				calculate_vesicles_within(intersections, save_list=False, name=name, chunking=True, num_chunks=num_chunks, sv=True))

	
		else:
			#no chunking, pass in vesicle files directly as parameter
			print("no chunking")

			print("---LV---")
			print(f"LV within neuron {name} & within {threshold}nm of another neuron: ", 
				calculate_vesicles_within(intersections, vesicles=cache["lv"], save_list=True, name=name), "\n")

			print("---SV---")
			print(f"SV within neuron {name} & within {threshold}nm of another neuron: ", 
				calculate_vesicles_within(intersections, vesicles=cache["sv"], name=name))

	
		







