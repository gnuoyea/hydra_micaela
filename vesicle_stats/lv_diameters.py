import numpy as np
import re
import scipy.stats as stats
import statistics
import pandas as pd
import ast
import os

#for finding thresholds for near neuron counts

#updated to include new vesicles types merging
#extract vesicle stats among all metadata to find overall mean and standard dev

names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
		"RGC2", "KM4", "NET12", "NET10", "NET11", "PN7", "SHL18", 
		"SHL24", "SHL28", "RGC7", "SHL17"]


cache = {} #for SV types mapping dictionaries
for name in names_20:
	dictionary = dict()
	cache[f"{name}"] = dictionary

#update for new exported com map directories + lv only
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


if __name__ == "__main__":
	results = []
	#combined_result = #create np array for all neurons counts

	names = names_20

	#initialize lists to store values for all neurons combined
	
	all_lv_diameters = []
	all_cv_diameters = []
	all_dv_diameters = []
	all_dvh_diameters = []

	for name in names:
		#lv_label_to_type = lv_labels_dict(f"types_lists/{name}_types.txt") #for type info

		print(f"---diameter stats for {name}---")
		lv_dictionary = read_txt_to_dict(name, "lv")

		####initialize stuff for LV diameters

		####LV diameters

		old_labels_dict = lv_labels_dict(f"/home/rothmr/hydra/types/old_types/{name}_types.txt")

		add_data = ["NET11", "RGC7", "SHL24", "SHL28", "NET10", "SHL18", "SHL17"]
		adding = False
		if(name in add_data):
			new_labels_dict = lv_labels_dict(f"/home/rothmr/hydra/types/new_types/new_v0+v2/{name}_lv_label.txt")
			adding = True

		####types diameters
		cv_diameters = []
		dv_diameters = []
		dvh_diameters = []
		print(f"length of dict for {name}", len(lv_dictionary.items()))
		for com,attributes in lv_dictionary.items():
			label = int(attributes[1][3:]) #just the number; could be overlap

			new_vesicle = False
			if(len(attributes)==7):
				new_vesicle = True

			if((label in old_labels_dict.keys()) or (adding and label in new_labels_dict.keys())):
				diameter = attributes[3]*2

				if(len(attributes)==6): #old vesicle
					subtype = old_labels_dict[label]

				elif(len(attributes)==7): #new vesicle
					if(label in new_labels_dict.keys()):
						subtype = new_labels_dict[label]
					else:
						print("error - new label not in dict")
				else:
					print("error - invalid attributes length = ", len(attributes))

				if(subtype==1):
					cv_diameters.append(diameter)

				if(subtype==2):
					dv_diameters.append(diameter)

				if(subtype==3):
					dvh_diameters.append(diameter)

			else:
				#if(adding and label in new_labels_dict.keys()): 
				if(new_vesicle):
					print(f"error - (new) label not in any dict: {label}")
				else:
					print(f"error - (old) label not in any dict: {label}")


		all_cv_diameters.extend(cv_diameters)
		all_dv_diameters.extend(dv_diameters)
		all_dvh_diameters.extend(dvh_diameters)

		num_lv_this_neuron = len(cv_diameters) + len(dv_diameters) + len(dvh_diameters)
		print(f"{name} total num LV (minus extraneous and unclassified): ", num_lv_this_neuron)


	####now do calculations for all neurons combined - alr done in sheet
	print("---all neurons combined stats---")

	####diameters
	all_lv_diameters = all_cv_diameters + all_dv_diameters + all_dvh_diameters
	all_lv_diameters = np.array(all_lv_diameters)
	mean = np.mean(all_lv_diameters)
	stdev = statistics.stdev(all_lv_diameters)
	sem = stats.sem(all_lv_diameters)
	n = len(all_lv_diameters)
	threshold = mean + (2*stdev) #725.1193604203999 nm
	print(f"ALL NEURONS - LV mean (diameters): {mean}, LV stdev: {stdev}, LV sem: {sem}, LV n: {n}, LV Threshold: {threshold}")
	results.append(["LV", "TOTAL", "Diameter", mean, stdev, sem, n, threshold])
	

	all_cv_diameters = np.array(all_cv_diameters)
	mean = np.mean(all_cv_diameters)
	stdev = statistics.stdev(all_cv_diameters)
	sem = stats.sem(all_cv_diameters)
	n = len(all_cv_diameters)
	threshold = mean + (2*stdev) #
	print(f"ALL NEURONS - CV mean (diameters): {mean}, CV stdev: {stdev}, CV sem: {sem}, CV n: {n}, CV Threshold: {threshold}")
	results.append(["CV", "TOTAL", "Diameter", mean, stdev, sem, n, threshold])

	all_dv_diameters = np.array(all_dv_diameters)
	mean = np.mean(all_dv_diameters)
	stdev = statistics.stdev(all_dv_diameters)
	sem = stats.sem(all_dv_diameters)
	n = len(all_dv_diameters)
	threshold = mean + (2*stdev) #
	print(f"ALL NEURONS - DV mean (diameters): {mean}, DV stdev: {stdev}, DV sem: {sem}, DV n: {n}, DV Threshold: {threshold}")
	results.append(["DV", "TOTAL", "Diameter", mean, stdev, sem, n, threshold])

	all_dvh_diameters = np.array(all_dvh_diameters)
	mean = np.mean(all_dvh_diameters)
	stdev = statistics.stdev(all_dvh_diameters)
	sem = stats.sem(all_dvh_diameters)
	n = len(all_dvh_diameters)
	threshold = mean + (2*stdev) #
	print(f"ALL NEURONS - DVH mean (diameters): {mean}, DVH stdev: {stdev}, DVH sem: {sem}, DVH n: {n}, DVH Threshold: {threshold}")
	results.append(["DVH", "TOTAL", "Diameter", mean, stdev, sem, n, threshold])



	df = pd.DataFrame(results, columns=["Type", "Neuron", "Measurement", "Mean", "Std Dev", "SEM", "N", "Near neuron threshold"])
	df.to_excel("/home/rothmr/hydra/sheet_exports/lv_diameters.xlsx", index=False)
	print("export done")
































