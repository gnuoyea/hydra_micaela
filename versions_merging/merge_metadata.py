import numpy as np
import h5py
import math
import re
import os
from scipy import spatial

#merging new vesicle info into metadata, accounting for possible overlaps and different types mapping data

names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
			"RGC2", "KM4", "SHL17", "NET12", "NET10", "NET11", "PN7", "SHL18", 
			"SHL24", "SHL28", "RGC7"]

cache = {}

def load_data(name):
	print(f"begin loading {name}")
	path = f'/projects/weilab/dataset/hydra/results_0408/vesicle_big_{name}_30-32-32.h5'
	with h5py.File(path, 'r') as f:

		cache["LV"] = f["main"][:]
		print("loaded ", name)

def read_txt_to_dict(file_path):
	results_dict = {}
	with open(file_path, 'r') as file:
		for line in file:
			key_string, value_string = line.split(": ")
			value_string = re.sub(r'\b(lv_\d+)\b', r"'\1'", value_string) #change into string literals
			value_string = re.sub(r'\b(sv_\d+)\b', r"'\1'", value_string) #same for sv

			coords_str = key_string.replace("[", "").replace("]", "").strip()
			coords_list=coords_str.split()
			coords = [float(coord) for coord in coords_list]

			attributes=eval(value_string)
			results_dict[tuple(coords)]=list(attributes)

	return results_dict


def find_extra_vesicles(name):
	path = f"/projects/weilab/dataset/hydra/results_0408/extra_patch/vesicle_big-bbs_{name}_30-8-8.h5"
	with h5py.File(path, "r") as f:
		data =  np.array(f["main"]).astype(int)

	extra_vesicles = [line[0] for line in data]
	return extra_vesicles



if __name__ == "__main__":
	voxel_dims = [30,8,8]

	same = ["KM4","SHL55","KR5","KR11","PN3","KR10","NET12","KR4","KR6","PN7","RGC2","LUX2"] #all neurons that has no change from before
	same = [] #we are done exporting

	for name in same:
		print(f"processing {name}")
		#no change to dicts
		#copy over metadata from before (old_meta folder) - keep everything except densities
		old_lv_meta_path = f'/home/rothmr/hydra/meta/old_meta/lv/{name}_lv_com_mapping.txt'
		old_sv_meta_path = f'/home/rothmr/hydra/meta/old_meta/sv/{name}_sv_com_mapping.txt'
		new_lv_meta_path = f'/home/rothmr/hydra/meta/new_meta/{name}_lv_com_mapping.txt'
		new_sv_meta_path = f'/home/rothmr/hydra/meta/new_meta/{name}_sv_com_mapping.txt'
		#create dicts for both! and update as we go
		lv_meta_dict = read_txt_to_dict(old_lv_meta_path)
		sv_meta_dict = read_txt_to_dict(old_sv_meta_path)

		#lv only kdtree - raw density values
		#in each step - append to the attributes list in the dict
		lv_com_list = []
		for com, attributes in lv_meta_dict.items():
			physical_com = np.array([com[0] * voxel_dims[0],com[1] * voxel_dims[1],com[2] * voxel_dims[2]])
			lv_com_list.append(physical_com)

		print("total num of LV: ", len(lv_com_list))
		lv_com_array = np.array(lv_com_list)

		# Build the KDTree and query neighbors within kd_radius (nm)
		kd_radius = 500 #nm
		tree = spatial.KDTree(lv_com_array)
		neighbors = tree.query_ball_tree(tree, kd_radius)
		frequency = np.array([len(n) for n in neighbors])
		density = frequency / (kd_radius ** 2)

		for i, com in enumerate(lv_meta_dict.keys()): #use original COMs not physical units
			print(i)
			lv_no_densities = lv_meta_dict[tuple(com)][0:4] #take out densities from before
			lv_no_densities.append(density[i])
			lv_meta_dict[tuple(com)] = lv_no_densities

		#####

		#loop thru ALL coms and add all to a kdtree - raw density values
		#make sv com list
		sv_com_list = []
		for com, attributes in sv_meta_dict.items():
			physical_com = np.array([com[0] * voxel_dims[0],com[1] * voxel_dims[1],com[2] * voxel_dims[2]])
			sv_com_list.append(physical_com)

		print("total num of LV+SV: ", len(sv_com_list) + len(lv_com_list))
		sv_com_array = np.array(sv_com_list)

		combined_com_list = lv_com_list + sv_com_list
		combined_com_array = np.array(combined_com_list)
		offset = len(lv_com_list)

		#kdtree for ALL
		tree = spatial.KDTree(combined_com_array)
		kd_radius = 500 #nm
		neighbors = tree.query_ball_tree(tree, kd_radius)
		frequency = np.array([len(n) for n in neighbors])
		density = frequency / (kd_radius ** 2)

		combined_keys_list = list(lv_meta_dict.keys()) + list(sv_meta_dict.keys()) #use original COMs not physical

		for i, com in enumerate(combined_keys_list): #0 index
			print(i)
			if i<offset:
				lv_meta_dict[tuple(com)].append(density[i]) #keep density from lv only
			else: #i<=offset / rest of the densities
				sv_no_densities = sv_meta_dict[tuple(com)][0:4] #take out densities from before
				sv_no_densities.append(density[i])
				sv_meta_dict[tuple(com)] = sv_no_densities

		####

		#write to new lv meta path for this neuron - generate txt (try to make same format)
		with open(new_lv_meta_path, "w") as f:
			for com, attributes in lv_meta_dict.items():
				f.write(f"{com}: ('{attributes[0]}', {attributes[1]}, {attributes[2]}, {attributes[3]}, {attributes[4]}, {attributes[5]})\n")
			print(f"LV meta for {name} successfully exported")

		#write to new sv meta path for this neuron - generate txt (try to make same format)
		#only 5 attributes bc no sv only kdtree
		with open(new_sv_meta_path, "w") as f:
			for com, attributes in sv_meta_dict.items():
				f.write(f"{com}: ('{attributes[0]}', {attributes[1]}, {attributes[2]}, {attributes[3]}, {attributes[4]})\n")
			print(f"SV meta for {name} successfully exported")



	add_data = ["SHL17"]
	running = ["NET11", "RGC7", "SHL24", "SHL28", "NET10", "SHL18"]
	for name in add_data:
		print(f"processing {name}")

		#init everything
		old_lv_meta_path = f'/home/rothmr/hydra/meta/old_meta/lv/{name}_lv_com_mapping.txt'
		old_sv_meta_path = f'/home/rothmr/hydra/meta/old_meta/sv/{name}_sv_com_mapping.txt'
		new_lv_meta_path = f'/home/rothmr/hydra/meta/new_meta/{name}_lv_com_mapping.txt'
		new_sv_meta_path = f'/home/rothmr/hydra/meta/new_meta/{name}_sv_com_mapping.txt'
		#create dicts for both! and update as we go
		lv_meta_dict = read_txt_to_dict(old_lv_meta_path)
		sv_meta_dict = read_txt_to_dict(old_sv_meta_path)

		load_data(name) #load in LV data only (v2)
		#cache["LV"] now stores new LV data

		to_add = find_extra_vesicles(name) #list of segids

		#error check that nothing from to_add is already in lv_meta_dict (overlaps...)
		for segid in to_add:
			if segid in lv_meta_dict.keys():
				print(f"error - not a new segid: {segid}")
		print("error checking done")
		print("num of extra vesicles: ", len(to_add))

		unique_labels = np.unique(cache["LV"])

		stats = {}
		counter = 0
		for label in unique_labels:
			if label in to_add:
				counter+=1
				print(f"extra vesicle #{counter}")
				if label == 0:
					continue  # Skip background
				mask = (cache["LV"] == label)
				coords = np.argwhere(mask)
				del mask
				if coords.size == 0:
					continue
				sum_coords = coords.sum(axis=0)
				count = coords.shape[0]
				if label in stats:
					stats[label]['sum'] += sum_coords
					stats[label]['count'] += count
				else:
					stats[label] = {'sum': sum_coords, 'count': count}
		print("checkpoint 1")

		extra_meta = {}
		for label, stats in stats.items():
			com = stats['sum'] / stats['count']
			volume_nm = stats['count'] * (voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
			radius_nm = math.sqrt((stats['count'] * (voxel_dims[1] * voxel_dims[2])) / math.pi)
			extra_meta[label] = {
				'com': com,
				'volume_nm': volume_nm,
				'radius_nm': radius_nm
				}

		print("checkpoint 2")

		#add everything from extra_meta into lv_meta_dict, and add new attribute thing for each
		for label, info in extra_meta.items():
			com = info["com"]
			if(tuple(com) in lv_meta_dict.keys()):
				print("com overlap")
			else:
				lv_meta_dict[tuple(com)] = ['lv', f"lv_{label}", info['volume_nm'], info['radius_nm'], "new"]
				#add an EXTRA item to attributes list here, keep track later too


		#lv only kdtree - raw density values
		#in each step - append to the attributes list in the dict
		lv_com_list = []
		for com, attributes in lv_meta_dict.items():
			physical_com = np.array([com[0] * voxel_dims[0],com[1] * voxel_dims[1],com[2] * voxel_dims[2]])
			lv_com_list.append(physical_com)

		print("total num of LV: ", len(lv_com_list))
		lv_com_array = np.array(lv_com_list)

		# Build the KDTree and query neighbors within kd_radius (nm)
		kd_radius = 500
		tree = spatial.KDTree(lv_com_array)
		neighbors = tree.query_ball_tree(tree, kd_radius)
		frequency = np.array([len(n) for n in neighbors])
		density = frequency / (kd_radius ** 2)

		for i, com in enumerate(lv_meta_dict.keys()): #use original COMs not physical units
			if(len(lv_meta_dict[com])==5): #new vesicle - attributes list is longer - 5 instead of 4
				lv_no_densities = lv_meta_dict[tuple(com)][0:4] #take out densities from before and also take out "new"
				lv_no_densities.append(density[i])
				lv_no_densities.append("new") #reappend "new" marker
				lv_meta_dict[tuple(com)] = lv_no_densities
			else: #NOT new vesicle
				lv_no_densities = lv_meta_dict[tuple(com)][0:4] #take out densities from before
				lv_no_densities.append(density[i])
				lv_meta_dict[tuple(com)] = lv_no_densities

		#####

		#loop thru ALL coms and add all to a kdtree - raw density values
		#make sv com list
		sv_com_list = []
		for com, attributes in sv_meta_dict.items():
			physical_com = np.array([com[0] * voxel_dims[0],com[1] * voxel_dims[1],com[2] * voxel_dims[2]])
			sv_com_list.append(physical_com)

		print("total num of LV+SV: ", len(sv_com_list) + len(lv_com_list))
		sv_com_array = np.array(sv_com_list)

		combined_com_list = lv_com_list + sv_com_list
		combined_com_array = np.array(combined_com_list)
		offset = len(lv_com_list)

		#kdtree for ALL
		tree = spatial.KDTree(combined_com_array)
		kd_radius = 500 #nm
		neighbors = tree.query_ball_tree(tree, kd_radius)
		frequency = np.array([len(n) for n in neighbors])
		density = frequency / (kd_radius ** 2)

		combined_keys_list = list(lv_meta_dict.keys()) + list(sv_meta_dict.keys()) #use original COMs not physical

		for i, com in enumerate(combined_keys_list): #0 index
			if i<offset: #LV
				if(len(lv_meta_dict[com])==6): #new vesicle - attributes list is longer - 6 instead of 5
					lv_no_densities = lv_meta_dict[tuple(com)][0:5] #take out "new" marker
					lv_no_densities.append(density[i])
					lv_no_densities.append("new") #reappend "new" marker to updated
					lv_meta_dict[tuple(com)] = lv_no_densities
				else: #NOT new vesicle
					lv_meta_dict[tuple(com)].append(density[i]) #keep density from lv only

			else: #i<=offset / rest of the densities - SV
				sv_no_densities = sv_meta_dict[tuple(com)][0:4] #take out densities from before
				sv_no_densities.append(density[i])
				sv_meta_dict[tuple(com)] = sv_no_densities

		####

		#write to new lv meta path for this neuron - generate txt (try to make same format)
		with open(new_lv_meta_path, "w") as f:
			for com, attributes in lv_meta_dict.items():
				if(len(attributes)==7): #new vesicle marker
					f.write(f"{com}: ('{attributes[0]}', {attributes[1]}, {attributes[2]}, {attributes[3]}, {attributes[4]}, {attributes[5]}, {attributes[6]})\n")
				else: #not new vesicle
					f.write(f"{com}: ('{attributes[0]}', {attributes[1]}, {attributes[2]}, {attributes[3]}, {attributes[4]}, {attributes[5]})\n")
			print(f"LV meta for {name} successfully exported")

		#write to new sv meta path for this neuron - generate txt (try to make same format)
		#only 5 attributes bc no sv only kdtree
		with open(new_sv_meta_path, "w") as f:
			for com, attributes in sv_meta_dict.items():
				f.write(f"{com}: ('{attributes[0]}', {attributes[1]}, {attributes[2]}, {attributes[3]}, {attributes[4]})\n")
			print(f"SV meta for {name} successfully exported")


		

	remove_data = ["SHL20"]
	remove_data = [] #we are done exporting
	for name in remove_data:
		print(f"processing {name}")

		to_remove = [1606, 1608, 2340, 2610] #assume segids are uniform across v0-> v2

		old_lv_meta_path = f'/home/rothmr/hydra/meta/old_meta/lv/{name}_lv_com_mapping.txt'
		old_sv_meta_path = f'/home/rothmr/hydra/meta/old_meta/sv/{name}_sv_com_mapping.txt'
		new_lv_meta_path = f'/home/rothmr/hydra/meta/new_meta/{name}_lv_com_mapping.txt'
		new_sv_meta_path = f'/home/rothmr/hydra/meta/new_meta/{name}_sv_com_mapping.txt'
		#create dicts for both! and update as we go
		lv_meta_dict = read_txt_to_dict(old_lv_meta_path)
		sv_meta_dict = read_txt_to_dict(old_sv_meta_path)

		#REMOVE stuff from lv_meta_dict
		items_list = [(com,attributes) for com,attributes in lv_meta_dict.items()] #so change doesn't conflict with iter
		for element in items_list:
			com = element[0]
			attributes = element[1]

			segid = int(attributes[1][3:])
			if segid in to_remove:
				print(f"removing LV vesicle {segid} from {name}")
				del lv_meta_dict[tuple(com)]


		#lv only kdtree - raw density values
		#in each step - append to the attributes list in the dict
		lv_com_list = []
		for com, attributes in lv_meta_dict.items():
			physical_com = np.array([com[0] * voxel_dims[0],com[1] * voxel_dims[1],com[2] * voxel_dims[2]])
			lv_com_list.append(physical_com)

		print("total num of LV: ", len(lv_com_list))
		lv_com_array = np.array(lv_com_list)

		# Build the KDTree and query neighbors within kd_radius (nm)
		kd_radius = 500
		tree = spatial.KDTree(lv_com_array)
		neighbors = tree.query_ball_tree(tree, kd_radius)
		frequency = np.array([len(n) for n in neighbors])
		density = frequency / (kd_radius ** 2)

		for i, com in enumerate(lv_meta_dict.keys()): #use original COMs not physical units
			lv_no_densities = lv_meta_dict[tuple(com)][0:4] #take out densities from before
			lv_no_densities.append(density[i])
			lv_meta_dict[tuple(com)] = lv_no_densities

		#####

		#loop thru ALL coms and add all to a kdtree - raw density values
		#make sv com list
		sv_com_list = []
		for com, attributes in sv_meta_dict.items():
			physical_com = np.array([com[0] * voxel_dims[0],com[1] * voxel_dims[1],com[2] * voxel_dims[2]])
			sv_com_list.append(physical_com)

		print("total num of LV+SV: ", len(sv_com_list) + len(lv_com_list))
		sv_com_array = np.array(sv_com_list)

		combined_com_list = lv_com_list + sv_com_list
		combined_com_array = np.array(combined_com_list)
		offset = len(lv_com_list)

		#kdtree for ALL
		tree = spatial.KDTree(combined_com_array)
		kd_radius = 500 #nm
		neighbors = tree.query_ball_tree(tree, kd_radius)
		frequency = np.array([len(n) for n in neighbors])
		density = frequency / (kd_radius ** 2)

		combined_keys_list = list(lv_meta_dict.keys()) + list(sv_meta_dict.keys()) #use original COMs not physical

		for i, com in enumerate(combined_keys_list): #0 index
			if i<offset:
				lv_meta_dict[tuple(com)].append(density[i]) #keep density from lv only

			else: #i<=offset / rest of the densities
				sv_no_densities = sv_meta_dict[tuple(com)][0:4] #take out densities from before
				sv_no_densities.append(density[i])
				sv_meta_dict[tuple(com)] = sv_no_densities

		####

		#write to new lv meta path for this neuron - generate txt (try to make same format)
		with open(new_lv_meta_path, "w") as f:
			for com, attributes in lv_meta_dict.items():
				f.write(f"{com}: ('{attributes[0]}', {attributes[1]}, {attributes[2]}, {attributes[3]}, {attributes[4]}, {attributes[5]})\n")
			print(f"LV meta for {name} successfully exported")

		#write to new sv meta path for this neuron - generate txt (try to make same format)
		#only 5 attributes bc no sv only kdtree
		with open(new_sv_meta_path, "w") as f:
			for com, attributes in sv_meta_dict.items():
				f.write(f"{com}: ('{attributes[0]}', {attributes[1]}, {attributes[2]}, {attributes[3]}, {attributes[4]})\n")
			print(f"SV meta for {name} successfully exported")



		





