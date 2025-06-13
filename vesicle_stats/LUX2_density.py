import numpy as np
import h5py
import ast
import math
import re
import scipy.stats as stats

#calculate average density value for vesicles in the ball, vs vesicles in the rest of the neuron 
#normalize by number of vesicles in each of these regions

D0 = '/data/projects/weilab/dataset/hydra/results/'
D9 = ''


cache = {}
cache["dict"] = {}
cache["full_dict"] = {} #all attribute info

names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
			"RGC2", "KM4", "SHL17", "NET12", "NET10", "NET11", "PN7", "SHL18", 
			"SHL24", "SHL28", "RGC7"]

for name in names_20:
	dictionary = dict()
	cache[f"{name}"] = dictionary


#LUX2
def load_data():
	#load pointcloud for LUX2 from metadata
	sv_dictionary = read_txt_to_dict("LUX2", "sv")
	com_to_density = {key:value[-1] for key, value in sv_dictionary.items()}
	#add in lv
	lv_dictionary = read_txt_to_dict("LUX2", "lv")
	for key,value in lv_dictionary.items():
		com_to_density[key] = value[-1]

	print("loaded coms and densities") #from metadata
	cache["dict"]=com_to_density

	for key,value in sv_dictionary.items():
		cache["full_dict"][key] = value
	for key,value in lv_dictionary.items():
		cache["full_dict"][key] = value

	#load ball
	path = f"LUX2_ball_manual.h5"
	with h5py.File(path, "r") as f:
		cache["ball"] = (np.array(f["main"]) > 0).astype(int) #make boolean mask
		print("shape: ", f["main"].shape)
	print("mask loaded")

def read_txt_to_dict(name, which):
	if(which=="sv"):
		results_dict = {}
		file_path = f"{D9}sv/{name}_sv_com_mapping.txt" #only consider SV here
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

	if(which=="lv"):
		results_dict = {}
		file_path = f"{D9}lv/{name}_lv_com_mapping.txt" #only consider SV here
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


#return list of densities for vesicles within ball
def vesicles_within_ball():
	ball = cache["ball"]
	original_res = [30, 8, 8]
	ball_res = [30, 32, 32]
	mapping = com_to_density
	num_in_ball = 0

	within_ball_densities = []
	outside_ball_densities = []
	ves_within_ball = 0
	ves_outside_ball = 0

	LV_within_ball = 0
	CV_within_ball = 0
	DV_within_ball = 0
	DVH_within_ball = 0
	SV_within_ball = 0
	SDV_within_ball = 0
	SCV_within_ball = 0

	LV_outside_ball = 0
	CV_outside_ball = 0
	DV_outside_ball = 0
	DVH_outside_ball = 0
	SV_outside_ball = 0
	SDV_outside_ball = 0
	SCV_outside_ball = 0

	
	coms_list = mapping.keys() #extract from mapping
	for com in coms_list:
		voxels_com = [com[0]*(original_res[0]/ball_res[0]), com[1]*(original_res[1]/ball_res[1]), com[2]*(original_res[2]/ball_res[2])] #change based on res
		voxels_com = np.round(voxels_com).astype(int) #round for indexing in the mask
		if (ball[tuple(voxels_com)]!=0): #boolean
			within_ball_densities.append(com_to_density[com])
			ves_within_ball += 1

			if(cache["full_dict"][com][0])=="lv":
				LV_within_ball += 1
				label = (int)(cache["full_dict"][com][1][3:])
				lv_label_to_type = lv_labels_dict(f"types_lists/LUX2_types.txt")
				subtype = lv_label_to_type[label]

				if(subtype==1):
					CV_within_ball+=1
				if(subtype==2):
					DV_within_ball+=1
				if(subtype==3):
					DVH_within_ball+=1

			else: #sv
				SV_within_ball += 1
				label = (int)(cache["full_dict"][com][1][3:])
				sv_label_to_type = cache["LUX2"]
				subtype = sv_label_to_type[label]

				if(subtype==4):
					SDV_within_ball+=1
				if(subtype==5):
					SCV_within_ball+=1


		else:
			outside_ball_densities.append(com_to_density[com])
			ves_outside_ball += 1

			if(cache["full_dict"][com][0])=="lv":
				LV_outside_ball += 1
				label = (int)(cache["full_dict"][com][1][3:])
				lv_label_to_type = lv_labels_dict(f"types_lists/LUX2_types.txt")
				subtype = lv_label_to_type[label]

				if(subtype==1):
					CV_outside_ball+=1
				if(subtype==2):
					DV_outside_ball+=1
				if(subtype==3):
					DVH_outside_ball+=1

			else: #sv
				SV_outside_ball += 1
				label = (int)(cache["full_dict"][com][1][3:])
				sv_label_to_type = cache["LUX2"]
				subtype = sv_label_to_type[label]

				if(subtype==4):
					SDV_outside_ball+=1
				if(subtype==5):
					SCV_outside_ball+=1


	print()
	print("LV within ball: ", LV_within_ball)
	print("CV within ball: ", CV_within_ball)
	print("DV within ball: ", DV_within_ball)
	print("DVH within ball: ", DVH_within_ball)
	print("SV within ball: ", SV_within_ball)
	print("SDV within ball: ", SDV_within_ball)
	print("SCV within ball: ", SCV_within_ball)

	print()
	print("LV outside ball: ", LV_outside_ball)
	print("CV outside ball: ", CV_outside_ball)
	print("DV outside ball: ", DV_outside_ball)
	print("DVH outside ball: ", DVH_outside_ball)
	print("SV outside ball: ", SV_outside_ball)
	print("SDV outside ball: ", SDV_outside_ball)
	print("SCV outside ball: ", SCV_outside_ball)
	print()

	return within_ball_densities, outside_ball_densities

if __name__ == "__main__":
	name = "LUX2"

	with np.load("sv_types/SV_types.npz") as data:
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

	load_data()
	com_to_density = cache["dict"]
	print("total num vesicles", len(com_to_density))
	#filter out to only coms in the ball
	within_ball_densities, outside_ball_densities = vesicles_within_ball()

	print("average for within ball: ", np.mean(within_ball_densities))
	print("average for outside ball: ", np.mean(outside_ball_densities))

	print("SEM for within ball: ", stats.sem(within_ball_densities))
	print("SEM for outside ball: ", stats.sem(outside_ball_densities))

	print("num within ball: ", len(within_ball_densities))
	print("num outside ball: ", len (outside_ball_densities))

	#export within ball densities
	with open(f"LUX2/LUX2_within_ball_densities.txt", "w") as f:
		for density in within_ball_densities:
			f.write(str(density) + "\n")

	#export outside ball densities
	with open(f"LUX2/LUX2_outside_ball_densities.txt", "w") as f:
		for density in outside_ball_densities:
			f.write(str(density) + "\n")



	














