import h5py
import numpy as np
import pandas as pd 
import scipy.stats as stats
import re
import yaml
import edt
import ast
import gc
import argparse

#also install openpyxl into conda env

#refactoring counts.py - for the soma counts, pointcloud based approach - much faster
#export totals and (mean, sem, n) over all neurons?

#UPDATED - ONLY FOR LV

D0 = '/data/projects/weilab/dataset/hydra/results/'
D1 = '/home/rothmr/projects/stitched/stitched/' #stitched box files source dir

cache = {}


names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
			"RGC2", "KM4", "NET12", "NET10", "NET11", "PN7", "SHL18", 
			"SHL24", "SHL28", "RGC7", "SHL17"]

neuron_info = {"KR4": [1, 5], "KR5": [2, 2], "KR6": [4, 2], "SHL55": [3, 5], "PN3": [10, 3], "LUX2": [11, 5], "SHL20": [17, 5], "KR11": [8, 5], "KR10": [7, 4], 
			"RGC2": [6, 5], "KM4": [9, 5], "NET12": [15, 5], "NET10": [16, 1], "NET11": [14, 1], "PN7": [12, 5], "SHL18": [18, 1], 
			"SHL24": [5, 1], "SHL28": [20, 1], "RGC7": [13, 1], "SHL17": [19, 4]}



#updated version 04-12
def read_txt_to_dict(name, which):
	results_dict = {}

	file_path = f'/home/rothmr/hydra/meta/new_meta/{name}_{which}_com_mapping.txt'

	with open(file_path, 'r') as file:
		for line in file:
			key_string, value_string = line.split(": ")
			value_string = re.sub(r'\b(lv_\d+)\b', r"'\1'", value_string) #change into string literals
			value_string = re.sub(r'\b(sv_\d+)\b', r"'\1'", value_string) #same for sv

			coords_str = key_string.replace("[", "").replace("]", "").strip()
			coords_str = re.sub(r'[\(\)\,]', '', coords_str) #strip parens
			#print("cleaned: ", coords_str)

			coords_list=coords_str.split()
			coords = [float(coord) for coord in coords_list]

			#add quotes around "new" attribute
			value_string = re.sub(r'(?<!["\'])\bnew\b(?!["\'])', "'new'", value_string)
			attributes = ast.literal_eval(value_string)

			results_dict[tuple(coords)]=list(attributes)

	return results_dict


def lv_labels_dict(file_path):
	result_dict = {}
	with open(file_path, 'r') as file:
		for line in file:
			line = line.strip()
			pairs = line.strip('()').split('),(')

			for pair in pairs:
				key, value = pair.split(':')
				result_dict[int(key)] = int(value)

	return result_dict


def load_data(name): #load into cache
	with h5py.File(f"{D1}neuron_{name}_box_30-32-32.h5", 'r') as f: #stitched mask, contains everything w segids
		cache["box"] = np.array(f["main"]).astype(int)
	print("done loading neuron box")

	#cache["sv_mapping"] = read_txt_to_dict(name, "sv")
	cache["lv_mapping"] = read_txt_to_dict(name, "lv")



#outputs the num of vesicles within and the total
#9 means just everything
def calculate_vesicles_within(mask_name, which=9): #use data from cache - after loading
	#note that all data in the mapping txt file is in nm
	mask = cache[mask_name]

	#lv
	lv_mapping = cache["lv_mapping"] #com-> attributes metadata

	old_labels_dict = lv_labels_dict(f"/home/rothmr/hydra/types/old_types/{name}_types.txt")
	add_data = ["NET11", "RGC7", "SHL24", "SHL28", "NET10", "SHL18", "SHL17"]
	adding = False
	if(name in add_data):
		new_labels_dict = lv_labels_dict(f"/home/rothmr/hydra/types/new_types/new_v0+v2/{name}_lv_label.txt")
		adding = True

	total_num = 0
	num_in_mask = 0
	
	for com, attributes in lv_mapping.items():
		#find subtype
		subtype = 0
		label = int(attributes[1][3:]) #just the number; could be overlap

		if(adding): #if this is one of the neurons stuff has been added to
			new_vesicle = False
			if(len(attributes)==7):
				new_vesicle = True

			if((label in old_labels_dict.keys()) or (adding and label in new_labels_dict.keys())):
				if(new_vesicle):
					if(label in new_labels_dict.keys()):
						subtype = new_labels_dict[label]
					else:
						subtype = 0
				else:
					subtype = old_labels_dict[label] #should exist in here

		else: #neurons with no change
			#subtype, if exists, is just whatever is in the old file
			if(label in old_labels_dict.keys()):
				subtype = old_labels_dict[label]
			else: #if doesn't exist in the dict, set to 0
				subtype = 0
			

		##now, subtype should have the correct value
		if((which==9 and subtype!=0) or (subtype==which)): #if we are including everything, or if subtype matches up
			#then consider this vesicle
			total_num +=1

			if(name == "SHL17"):
				com = [com[0], com[1]+4000, com[2]]
			voxels_com = [com[0], com[1]/4, com[2]/4] #downsample since mask in 30-32-32
			voxels_com = np.round(voxels_com).astype(int) #round for indexing in the mask
			if (mask[tuple(voxels_com)]!=0): #boolean
				num_in_mask += 1

	print("total num: ", total_num)
	print("num in region: ", num_in_mask)


	return total_num, num_in_mask


def expand_mask(original_mask, threshold_nm, res): #threshold in physical units, res in xyz
	expanded_mask = np.copy(original_mask)

	print("begin distance transform for expansion")
	dt = edt.edt(1 - original_mask, anisotropy=(res[2],res[1],res[0])) #needs to be in xyz by default for edt
	print("end distance transform for expansion")
	doubled_perimeter = dt <= threshold_nm #all in nm
	expanded_mask[doubled_perimeter] = 1

	expanded_mask_binary = (expanded_mask>=1).astype(int)
	return expanded_mask_binary

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

def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

if __name__ == "__main__":

	neuron_dict = read_yml('/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt')
	names = names_20
	mask_res = [30, 32, 32]

	results = [] #for xlxs export

	all_percentages_in = [] #percentages in soma


	##### optional if running individually
	parser = argparse.ArgumentParser()
	parser.add_argument("name", type=str, help="neuron name")
	args = parser.parse_args()
	name = args.name

	names = [name]
    #####


	for name in names:
		#neuron id/type info
		neuron_id = neuron_info[name][0]
		neuron_type = neuron_info[name][1]


		load_data(name) #load neurons and mappings
		print(f"loaded data for {name}")
		nid = neuron_name_to_id(name)

		###
		t0 = 722.1704715 #all LV

		#extract regions of interest from mask
		neurons = cache["box"]

		neuron_only = np.zeros(neurons.shape, dtype=neurons.dtype)
		neuron_only[neurons==nid] = 1 #binary mask
		other_neurons = np.zeros(neurons.shape, dtype = neurons.dtype)
		other_neurons[(neurons!=nid) & (neurons!=0)] = 1 #binary mask


		print("LV")
		expanded_others = expand_mask(other_neurons, t0, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		print("size of intersection region: ", np.sum(cache["intersections"]))
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within("intersections") #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "LV", total_num, nn_lv, num_far, t0])

		print(f"{name} LV done \n")



		#cv
		print("CV")
		t1 = 771.2949661 #CV
		expanded_others = expand_mask(other_neurons, t1, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		print("size of intersection region: ", np.sum(cache["intersections"]))
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within("intersections", which=1) #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "CV", total_num, nn_lv, num_far, t1])
		print(f"{name} CV done \n")



		#dv
		print("DV")
		t2 = 561.7436736 #DV
		expanded_others = expand_mask(other_neurons, t2, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		print("size of intersection region: ", np.sum(cache["intersections"]))
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within("intersections", which=2) #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "DV", total_num, nn_lv, num_far, t2])
		print(f"{name} DV done \n")



		#dvh
		print("DVH")
		t3 = 858.8801088 #DVH
		expanded_others = expand_mask(other_neurons, t3, mask_res)
		print("mask expanding done")
		cache["intersections"] = mask_intersection((neuron_only, expanded_others))
		print("size of intersection region: ", np.sum(cache["intersections"]))
		print("mask intersection done")
		total_num, nn_lv = calculate_vesicles_within("intersections", which=3) #name of cache file
		num_far = total_num-nn_lv
		print("num outside of region: ", num_far)
		del expanded_others, cache["intersections"]
		gc.collect()
		#append to individual row of the results output
		results.append([name, neuron_id, neuron_type, "DVH", total_num, nn_lv, num_far, t3])

		print(f"{name} DVH done\n")




	df = pd.DataFrame(results, columns=["Neuron", "Neuron ID", "Neuron type", "Vesicle type", "Total num", "Near neuron", "Not near neuron", "Nearness threshold (nm)"])
	df.to_excel("/home/rothmr/hydra/sheet_exports/lv_near_counts.xlsx", index=False)
	print("export done")
	print("done")




















