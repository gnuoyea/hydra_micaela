import numpy as np
import re
import scipy.stats as stats
import pandas as pd
import ast
import argparse

#install openpyxl into conda env

#extract volume and diameter stats
#note - set up directories for list exports before running


sample_dir = '/home/rothmr/hydra/sample/'

names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
			"RGC2", "KM4", "SHL17", "NET12", "NET10", "NET11", "PN7", "SHL18", 
			"SHL24", "SHL28", "RGC7"]

cache = {} #for SV types mapping dictionaries
for name in names_20:
	dictionary = dict()
	cache[f"{name}"] = dictionary

def read_txt_to_dict(name, which):
	results_dict = {}

	if(name=="sample"):
		file_path = f'{sample_dir}sample_outputs/sample_com_mapping.txt'

	else:
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


#extract subtype of a given (LV) vesicle's COM for a given neuron; could be 0 if extraneous annotation / unclassified
def find_subtype(name, com):
	lv_mapping = read_txt_to_dict(name, "lv") #com-> attributes metadata
	attributes = lv_mapping[com]
	label = int(attributes[1][3:])

	if(name=="sample"):
		labels_dict_path = f"{sample_dir}sample_data/7-13_lv_label.txt"
	else:
		labels_dict_path = f"/home/rothmr/hydra/types/new_types/new_v0+v2/{name}_lv_label.txt"

	labels_dict = lv_labels_dict(labels_dict_path)

	if(label in labels_dict.keys()):
		subtype = labels_dict[label]
	else: #if doesn't exist in the dict, set to 0 (extraneous/unclassified)
		print("error - subtype not found")
		subtype = 0

	return subtype



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--which_neurons", type=str, help="all or sample?") #enter as "all" or "sample"
	args = parser.parse_args()
	which_neurons = args.which_neurons

	#ensure which_neurons is entered
	if(args.which_neurons is None):
		parser.error("error - must enter all or sample for --which_neurons")

	#set export dir, neuron list, and SV processing based on which arg chosen
	if(which_neurons=="sample"):
		D_exports = f'{sample_dir}sample_outputs/'
		sv_stats = False
		names = ["sample"]
	elif(which_neurons=="all"):
		D_exports = f"/home/rothmr/hydra/list_exports/"
		sv_stats = True
		names = names_20

	results = []
	
	#initialize lists to store values for all neurons combined
	all_lv_volumes = []
	all_cv_volumes = []
	all_dv_volumes = []
	all_dvh_volumes = []
	all_sv_volumes = []
	all_scv_volumes = []
	all_sdv_volumes = []
	all_all_volumes = []

	all_lv_diameters = []
	all_cv_diameters = []
	all_dv_diameters = []
	all_dvh_diameters = []
	all_sv_diameters = []
	all_scv_diameters = []
	all_sdv_diameters = []
	all_all_diameters = []


	#only if running all neurons, not sample
	if(sv_stats==True):
		#fill in SV dicts for all neurons
		with np.load("/home/rothmr/hydra/types/sv_types/SV_types_new.npz") as data: #UPDATED to new types mapping file for new classifications
			print("initializing SV types mapping dictionaries")
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
				cache[name] = {**cache[name], vesicle_id:vesicle_type} #update dict for the [name] neuron
				current+=1;

			print("initialized all SV types mapping dictionaries")

	for name in names:
		print(f"---volume stats for {name}---")
		####initialize stuff for LV volumes
		lv_dictionary = read_txt_to_dict(name, "lv")  #com-> attributes metadata

		lv_volumes = []
		lv_diameters = []
		cv_volumes = []
		cv_diameters = []
		dv_volumes = []
		dv_diameters = []
		dvh_volumes = []
		dvh_diameters = []


		for com, attributes in lv_dictionary.items():
			subtype = find_subtype(name, com)
			volume = attributes[2]
			diameter = attributes[3]*2 #since value in attributes is radius

			if(subtype==1):
				cv_volumes.append(volume)
				lv_volumes.append(volume)
				cv_diameters.append(diameter)
				lv_diameters.append(diameter)

			if(subtype==2):
				dv_volumes.append(volume)
				lv_volumes.append(volume)
				dv_diameters.append(diameter)
				lv_diameters.append(diameter)

			if(subtype==3):
				dvh_volumes.append(volume)
				lv_volumes.append(volume)
				dvh_diameters.append(diameter)
				lv_diameters.append(diameter)


		####LV volumes
		all_lv_volumes.extend(lv_volumes) #for later all neurons
		lv_volumes = np.array(lv_volumes)
		mean = np.mean(lv_volumes)
		sem = stats.sem(lv_volumes)
		n = len(lv_volumes)
		print(f"Total LV mean (volumes): {mean}, LV sem: {sem}, LV n: {n}")
		results.append(["LV", name, "Volume", mean, sem, n])
		#export to list
		with open(f"{D_exports}{name}_LV_volumes.txt", "w") as f:
			for vol in lv_volumes:
				f.write(str(vol) + "\n")

		all_cv_volumes.extend(cv_volumes) #for later all neurons
		cv_volumes = np.array(cv_volumes)
		mean = np.mean(cv_volumes)
		sem = stats.sem(cv_volumes)
		n = len(cv_volumes)
		print(f"CV mean (volumes): {mean}, CV sem: {sem}, CV n: {n}")
		results.append(["CV", name, "Volume", mean, sem, n])
		#export to list
		with open(f"{D_exports}{name}_CV_volumes.txt", "w") as f:
			for vol in cv_volumes:
				f.write(str(vol) + "\n")

		all_dv_volumes.extend(dv_volumes) #for later all neurons
		dv_volumes = np.array(dv_volumes)
		mean = np.mean(dv_volumes)
		sem = stats.sem(dv_volumes)
		n = len(dv_volumes)
		print(f"DV mean (volumes): {mean}, DV sem: {sem}, DV n: {n}")
		results.append(["DV", name, "Volume", mean, sem, n])
		#export to list
		with open(f"{D_exports}{name}_DV_volumes.txt", "w") as f:
			for vol in dv_volumes:
				f.write(str(vol) + "\n")

		all_dvh_volumes.extend(dvh_volumes) #for later all neurons
		dvh_volumes = np.array(dvh_volumes)
		mean = np.mean(dvh_volumes)
		sem = stats.sem(dvh_volumes)
		n = len(dvh_volumes)
		print(f"DVH mean (volumes): {mean}, DVH sem: {sem}, DVH n: {n}")
		results.append(["DVH", name, "Volume", mean, sem, n])
		#export to list
		with open(f"{D_exports}{name}_DVH_volumes.txt", "w") as f:
			for vol in dvh_volumes:
				f.write(str(vol) + "\n")

		if(sv_stats):
			####initialize stuff for SV volumes
			sv_label_to_type = cache[name] #for type info
			sv_dictionary = read_txt_to_dict(name, "sv")
			sv_label_to_vol = {int(value_list[1][3:]):value_list[2] for value_list in sv_dictionary.values()}

			####SV volumes
			sv_volumes = list(sv_label_to_vol.values())
			all_sv_volumes.extend(sv_volumes) #for later all neurons
			sv_volumes = np.array(sv_volumes)
			mean = np.mean(sv_volumes)
			sem = stats.sem(sv_volumes)
			n = len(sv_volumes)
			print(f"Total SV mean (volumes): {mean}, SV sem: {sem}, SV n: {n}")
			results.append(["SV", name, "Volume", mean, sem, n])
			with open(f"{D_exports}{name}_SV_volumes.txt", "w") as f:
				for vol in sv_volumes:
					f.write(str(vol) + "\n")

			sdv_volumes = list(value for label, value in sv_label_to_vol.items() if (label in sv_label_to_type and sv_label_to_type[label] == 4))
			all_sdv_volumes.extend(sdv_volumes) #for later all neurons
			sdv_volumes = np.array(sdv_volumes)
			mean = np.mean(sdv_volumes)
			sem = stats.sem(sdv_volumes)
			n = len(sdv_volumes)
			print(f"SDV mean (volumes): {mean}, SDV sem: {sem}, SDV n: {n}")
			results.append(["SDV", name, "Volume", mean, sem, n])
			with open(f"{D_exports}{name}_SDV_volumes.txt", "w") as f:
				for vol in sdv_volumes:
					f.write(str(vol) + "\n")

			scv_volumes = list(value for label, value in sv_label_to_vol.items() if (label in sv_label_to_type and sv_label_to_type[label] == 5))
			all_scv_volumes.extend(scv_volumes) #for later all neurons
			scv_volumes = np.array(scv_volumes)
			mean = np.mean(scv_volumes)
			sem = stats.sem(scv_volumes)
			n = len(scv_volumes)
			print(f"SCV mean (volumes): {mean}, SCV sem: {sem}, SCV n: {n}")
			results.append(["SCV", name, "Volume", mean, sem, n])
			with open(f"{D_exports}{name}_SCV_volumes.txt", "w") as f:
				for vol in scv_volumes:
					f.write(str(vol) + "\n")
		

		####before moving on - combine all types for this neuron and export stats
		all_volumes = []
		all_volumes.extend(lv_volumes)
		if(sv_stats):
			all_volumes.extend(sv_volumes)
		all_all_volumes.extend(all_volumes)
		all_volumes = np.array(all_volumes)
		mean = np.mean(all_volumes)
		sem = stats.sem(all_volumes)
		n = len(all_volumes)
		print(f"TOTAL mean (volumes): {mean}, TOTAL sem: {sem}, TOTAL n: {n}")
		results.append(["Total", name, "Volume", mean, sem, n])
		with open(f"{D_exports}{name}_all_volumes.txt", "w") as f:
			for vol in all_volumes:
				f.write(str(vol) + "\n")

		print(f"---diameter stats for {name}---")

		####LV diameters (lists already populated from before)
		all_lv_diameters.extend(lv_diameters)
		lv_diameters = np.array(lv_diameters)
		mean = np.mean(lv_diameters)
		sem = stats.sem(lv_diameters)
		n = len(lv_diameters)
		print(f"Total LV mean (diameters): {mean}, LV sem: {sem}, LV n: {n}")
		results.append(["LV", name, "Diameter", mean, sem, n])
		with open(f"{D_exports}{name}_LV_diameters.txt", "w") as f:
			for diam in lv_diameters:
				f.write(str(diam) + "\n")

		all_cv_diameters.extend(cv_diameters) #for later all neurons
		cv_diameters = np.array(cv_diameters)
		mean = np.mean(cv_diameters)
		sem = stats.sem(cv_diameters)
		n = len(cv_diameters)
		print(f"CV mean (diameters): {mean}, CV sem: {sem}, CV n: {n}")
		results.append(["CV", name, "Diameter", mean, sem, n])
		with open(f"{D_exports}{name}_CV_diameters.txt", "w") as f:
			for diam in cv_diameters:
				f.write(str(diam) + "\n")

		all_dv_diameters.extend(dv_diameters) #for later all neurons
		dv_diameters = np.array(dv_diameters)
		mean = np.mean(dv_diameters)
		sem = stats.sem(dv_diameters)
		n = len(dv_diameters)
		print(f"DV mean (diameters): {mean}, DV sem: {sem}, DV n: {n}")
		results.append(["DV", name, "Diameter", mean, sem, n])
		with open(f"{D_exports}{name}_DV_diameters.txt", "w") as f:
			for diam in dv_diameters:
				f.write(str(diam) + "\n")

		all_dvh_diameters.extend(dvh_diameters) #for later all neurons
		dvh_diameters = np.array(dvh_diameters)
		mean = np.mean(dvh_diameters)
		sem = stats.sem(dvh_diameters)
		n = len(dvh_diameters)
		print(f"DVH mean (diameters): {mean}, DVH sem: {sem}, DVH n: {n}")
		results.append(["DVH", name, "Diameter", mean, sem, n])
		with open(f"{D_exports}{name}_DVH_diameters.txt", "w") as f:
			for diam in dvh_diameters:
				f.write(str(diam) + "\n")

		if(sv_stats):
			####initialize stuff for SV diameters
			sv_label_to_diam = {int(value_list[1][3:]):value_list[3]*2 for value_list in sv_dictionary.values()}

			####SV diameters
			sv_diameters = list(sv_label_to_diam.values())
			all_sv_diameters.extend(sv_diameters) #for later all neurons
			sv_diameters = np.array(sv_diameters)
			mean = np.mean(sv_diameters)
			sem = stats.sem(sv_diameters)
			n = len(sv_diameters)
			print(f"Total SV mean (diameters): {mean}, SV sem: {sem}, SV n: {n}")
			results.append(["SV", name, "Diameter", mean, sem, n])
			with open(f"{D_exports}{name}_SV_diameters.txt", "w") as f:
				for diam in sv_diameters:
					f.write(str(diam) + "\n")

			sdv_diameters = list(value for label, value in sv_label_to_diam.items() if (label in sv_label_to_type and sv_label_to_type[label] == 4))
			all_sdv_diameters.extend(sdv_diameters) #for later all neurons
			sdv_diameters = np.array(sdv_diameters)
			mean = np.mean(sdv_diameters)
			sem = stats.sem(sdv_diameters)
			n = len(sdv_diameters)
			print(f"SDV mean (diameters): {mean}, SDV sem: {sem}, SDV n: {n}")
			results.append(["SDV", name, "Diameter", mean, sem, n])
			with open(f"{D_exports}{name}_SDV_diameters.txt", "w") as f:
				for diam in sdv_diameters:
					f.write(str(diam) + "\n")

			scv_diameters = list(value for label, value in sv_label_to_diam.items() if (label in sv_label_to_type and sv_label_to_type[label] == 5))
			all_scv_diameters.extend(scv_diameters) #for later all neurons
			scv_diameters = np.array(scv_diameters)
			mean = np.mean(scv_diameters)
			sem = stats.sem(scv_diameters)
			n = len(scv_diameters)
			print(f"SCV mean (diameters): {mean}, SCV sem: {sem}, SCV n: {n}")
			results.append(["SCV", name, "Diameter", mean, sem, n])
			with open(f"{D_exports}{name}_SCV_diameters.txt", "w") as f:
				for diam in scv_diameters:
					f.write(str(diam) + "\n")

		####before moving on - combine all types for this neuron and export stats
		all_diameters = []
		all_diameters.extend(lv_diameters)
		if(sv_stats):
			all_diameters.extend(sv_diameters)
		all_all_diameters.extend(all_diameters)
		all_diameters = np.array(all_diameters)
		mean = np.mean(all_diameters)
		sem = stats.sem(all_diameters)
		n = len(all_diameters)
		print(f"TOTAL mean (diameters): {mean}, TOTAL sem: {sem}, TOTAL n: {n}")
		results.append(["Total", name, "Diameter", mean, sem, n])
		with open(f"{D_exports}{name}_all_diameters.txt", "w") as f:
			for diam in all_diameters:
				f.write(str(diam) + "\n")

		print()



	####now do calculations for all neurons combined - alr done in sheet
	print("---all neurons combined stats---")

	####volumes
	all_lv_volumes = np.array(all_lv_volumes)
	mean = np.mean(all_lv_volumes)
	sem = stats.sem(all_lv_volumes)
	n = len(all_lv_volumes)
	print(f"ALL NEURONS - LV mean (volumes): {mean}, LV sem: {sem}, LV n: {n}")
	results.append(["LV", "TOTAL", "Volume", mean, sem, n])
	with open(f"{D_exports}all_LV_volumes.txt", "w") as f:
		for vol in all_lv_volumes:
			f.write(str(vol) + "\n")

	all_cv_volumes = np.array(all_cv_volumes)
	mean = np.mean(all_cv_volumes)
	sem = stats.sem(all_cv_volumes)
	n = len(all_cv_volumes)
	print(f"ALL NEURONS - CV mean (volumes): {mean}, CV sem: {sem}, CV n: {n}")
	results.append(["CV", "TOTAL", "Volume", mean, sem, n])
	with open(f"{D_exports}all_CV_volumes.txt", "w") as f:
		for vol in all_cv_volumes:
			f.write(str(vol) + "\n")

	all_dv_volumes = np.array(all_dv_volumes)
	mean = np.mean(all_dv_volumes)
	sem = stats.sem(all_dv_volumes)
	n = len(all_dv_volumes)
	print(f"ALL NEURONS - DV mean (volumes): {mean}, DV sem: {sem}, DV n: {n}")
	results.append(["DV", "TOTAL", "Volume", mean, sem, n])
	with open(f"{D_exports}all_DV_volumes.txt", "w") as f:
		for vol in all_dv_volumes:
			f.write(str(vol) + "\n")

	all_dvh_volumes = np.array(all_dvh_volumes)
	mean = np.mean(all_dvh_volumes)
	sem = stats.sem(all_dvh_volumes)
	n = len(all_dvh_volumes)
	print(f"ALL NEURONS - DVH mean (volumes): {mean}, DVH sem: {sem}, DVH n: {n}")
	results.append(["DVH", "TOTAL", "Volume", mean, sem, n])
	with open(f"{D_exports}all_DVH_volumes.txt", "w") as f:
		for vol in all_dvh_volumes:
			f.write(str(vol) + "\n")

	if(sv_stats):
		all_sv_volumes = np.array(all_sv_volumes)
		mean = np.mean(all_sv_volumes)
		sem = stats.sem(all_sv_volumes)
		n = len(all_sv_volumes)
		print(f"ALL NEURONS - SV mean (volumes): {mean}, SV sem: {sem}, SV n: {n}")
		results.append(["SV", "TOTAL", "Volume", mean, sem, n])
		with open(f"{D_exports}all_SV_volumes.txt", "w") as f:
			for vol in all_sv_volumes:
				f.write(str(vol) + "\n")

		all_scv_volumes = np.array(all_scv_volumes)
		mean = np.mean(all_scv_volumes)
		sem = stats.sem(all_scv_volumes)
		n = len(all_scv_volumes)
		print(f"ALL NEURONS - SCV mean (volumes): {mean}, SCV sem: {sem}, SCV n: {n}")
		results.append(["SCV", "TOTAL", "Volume", mean, sem, n])
		with open(f"{D_exports}all_SCV_volumes.txt", "w") as f:
			for vol in all_scv_volumes:
				f.write(str(vol) + "\n")

		all_sdv_volumes = np.array(all_sdv_volumes)
		mean = np.mean(all_sdv_volumes)
		sem = stats.sem(all_sdv_volumes)
		n = len(all_sdv_volumes)
		print(f"ALL NEURONS - SDV mean (volumes): {mean}, SDV sem: {sem}, SDV n: {n}")
		results.append(["SDV", "TOTAL", "Volume", mean, sem, n])
		with open(f"{D_exports}all_SDV_volumes.txt", "w") as f:
			for vol in all_sdv_volumes:
				f.write(str(vol) + "\n")

	all_all_volumes = np.array(all_all_volumes)
	mean = np.mean(all_all_volumes)
	sem = stats.sem(all_all_volumes)
	n = len(all_all_volumes)
	print(f"ALL NEURONS - TOTAL mean (volumes): {mean}, TOTAL sem: {sem}, TOTAL n: {n}")
	results.append(["Total", "TOTAL", "Volume", mean, sem, n])
	with open(f"{D_exports}all_all_volumes.txt", "w") as f:
		for vol in all_all_volumes:
			f.write(str(vol) + "\n")

	####diameters
	all_lv_diameters = np.array(all_lv_diameters)
	mean = np.mean(all_lv_diameters)
	sem = stats.sem(all_lv_diameters)
	n = len(all_lv_diameters)
	print(f"ALL NEURONS - LV mean (diameters): {mean}, LV sem: {sem}, LV n: {n}")
	results.append(["LV", "TOTAL", "Diameter", mean, sem, n])
	with open(f"{D_exports}all_LV_diameters.txt", "w") as f:
		for diam in all_lv_diameters:
			f.write(str(diam) + "\n")

	all_cv_diameters = np.array(all_cv_diameters)
	mean = np.mean(all_cv_diameters)
	sem = stats.sem(all_cv_diameters)
	n = len(all_cv_diameters)
	print(f"ALL NEURONS - CV mean (diameters): {mean}, CV sem: {sem}, CV n: {n}")
	results.append(["CV", "TOTAL", "Diameter", mean, sem, n])
	with open(f"{D_exports}all_CV_diameters.txt", "w") as f:
		for diam in all_cv_diameters:
			f.write(str(diam) + "\n")

	all_dv_diameters = np.array(all_dv_diameters)
	mean = np.mean(all_dv_diameters)
	sem = stats.sem(all_dv_diameters)
	n = len(all_dv_diameters)
	print(f"ALL NEURONS - DV mean (diameters): {mean}, DV sem: {sem}, DV n: {n}")
	results.append(["DV", "TOTAL", "Diameter", mean, sem, n])
	with open(f"{D_exports}all_DV_diameters.txt", "w") as f:
		for diam in all_dv_diameters:
			f.write(str(diam) + "\n")

	all_dvh_diameters = np.array(all_dvh_diameters)
	mean = np.mean(all_dvh_diameters)
	sem = stats.sem(all_dvh_diameters)
	n = len(all_dvh_diameters)
	print(f"ALL NEURONS - DVH mean (diameters): {mean}, DVH sem: {sem}, DVH n: {n}")
	results.append(["DVH", "TOTAL", "Diameter", mean, sem, n])
	with open(f"{D_exports}all_DVH_diameters.txt", "w") as f:
		for diam in all_dvh_diameters:
			f.write(str(diam) + "\n")

	if(sv_stats):
		all_sv_diameters = np.array(all_sv_diameters)
		mean = np.mean(all_sv_diameters)
		sem = stats.sem(all_sv_diameters)
		n = len(all_sv_diameters)
		print(f"ALL NEURONS - SV mean (diameters): {mean}, SV sem: {sem}, SV n: {n}")
		results.append(["SV", "TOTAL", "Diameter", mean, sem, n])
		with open(f"{D_exports}all_SV_diameters.txt", "w") as f:
			for diam in all_sv_diameters:
				f.write(str(diam) + "\n")

		all_scv_diameters = np.array(all_scv_diameters)
		mean = np.mean(all_scv_diameters)
		sem = stats.sem(all_scv_diameters)
		n = len(all_scv_diameters)
		print(f"ALL NEURONS - SCV mean (diameters): {mean}, SCV sem: {sem}, SCV n: {n}")
		results.append(["SCV", "TOTAL", "Diameter", mean, sem, n])
		with open(f"{D_exports}all_SCV_diameters.txt", "w") as f:
			for diam in all_scv_diameters:
				f.write(str(diam) + "\n")

		all_sdv_diameters = np.array(all_sdv_diameters)
		mean = np.mean(all_sdv_diameters)
		sem = stats.sem(all_sdv_diameters)
		n = len(all_sdv_diameters)
		print(f"ALL NEURONS - SDV mean (diameters): {mean}, SDV sem: {sem}, SDV n: {n}")
		results.append(["SDV", "TOTAL", "Diameter", mean, sem, n])
		with open(f"{D_exports}all_SDV_diameters.txt", "w") as f:
			for diam in all_sdv_diameters:
				f.write(str(diam) + "\n")

	all_all_diameters = np.array(all_all_diameters)
	mean = np.mean(all_all_diameters)
	sem = stats.sem(all_all_diameters)
	n = len(all_all_diameters)
	print(f"ALL NEURONS - TOTAL mean (diameters): {mean}, TOTAL sem: {sem}, TOTAL n: {n}")
	results.append(["Total", "TOTAL", "Diameter", mean, sem, n])
	with open(f"{D_exports}all_all_diameters.txt", "w") as f:
		for diam in all_all_diameters:
			f.write(str(diam) + "\n")


	df = pd.DataFrame(results, columns=["Type", "Neuron", "Measurement", "Mean", "SEM", "N"])
	df.to_excel(f"{D_exports}vesicle_stats.xlsx", index=False)
	print("export done")





























