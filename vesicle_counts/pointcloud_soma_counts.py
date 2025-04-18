import h5py
import numpy as np
import pandas as pd 
import scipy.stats as stats
import re

#refactoring counts.py - for the soma counts, pointcloud based approach
#export totals and (mean, sem, n) over all neurons?

D0 = '/data/projects/weilab/dataset/hydra/results/'
D9 = 'skibidi/sv/'

cache = {} #stores "soma", "sv_mapping"

names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
			"RGC2", "KM4", "SHL17", "NET12", "NET10", "NET11", "PN7", "SHL18", 
			"SHL24", "SHL28", "RGC7"]

#this is based on 02-17 old version of metadata
def read_txt_to_dict(name):
	results_dict = {}
	file_path = f"{D9}{name}_sv_com_mapping.txt" #only consider SV here
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

def load_data(name): #load into cache
	#soma
	with h5py.File(f"{D0}soma_{name}_30-32-32.h5") as f:
		cache["soma"] = (np.array(f["main"]) > 0).astype(int) #make boolean mask
	#neuron
	with h5py.File(f"{D0}neuron_{name}_30-32-32.h5") as f:
		cache["neuron"] = (np.array(f["main"]) > 0).astype(int) #make boolean mask
	#sv_mapping
	cache["sv_mapping"] = read_txt_to_dict(name)



#outputs the num of vesicles within and the total
def calculate_vesicles_within(mask_res): #use data from cache - after loading
	#note that all data in the mapping txt file is in nm
	soma = cache["soma"]
	mapping = cache["sv_mapping"]

	total_num_sv = len(mapping)
	num_in_soma = 0
	
	coms_list = mapping.keys() #extract from mapping
	for com in coms_list:
		if(name == "SHL17"):
			com = [com[0], com[1]+4000, com[2]]
			print(com)
		voxels_com = [com[0], com[1]/4, com[2]/4] #downsample since mask in 30-32-32
		voxels_com = np.round(voxels_com).astype(int) #round for indexing in the mask
		if (soma[tuple(voxels_com)]!=0): #boolean
			num_in_soma+=1

	print("num in soma: ", num_in_soma)
	return total_num_sv, num_in_soma


if __name__ == "__main__":
	names =  names_20
	mask_res = [30, 32, 32]

	results = [] #for xlxs export

	all_percentages_in = [] #percentages in soma

	for name in names:
		load_data(name) #loads soma and mapping into cache
		print(f"loaded data for {name}")
		total_num_sv, num_in_soma = calculate_vesicles_within(mask_res)
		num_outside_soma = total_num_sv - num_in_soma
		percent_in_soma = num_in_soma/total_num_sv * 100
		percent_outside_soma = 100-percent_in_soma

		soma = cache["soma"]
		neuron = cache["neuron"]

		#we can do this bc both masks are in the same res (30-32-32)
		soma_vol = np.sum(soma)
		neuron_vol = np.sum(neuron)
		soma_size = (soma_vol/neuron_vol)*100

		#append to individual row of the results output
		results.append([name, total_num_sv, num_in_soma, num_outside_soma, percent_in_soma, percent_outside_soma, soma_size])

		all_percentages_in.append(percent_in_soma)

		print(f"{name} done")

	results.append([]) #blank line to separate
	

	#calculate mean, sem, n - this is of the list of percentages, across all neurons?
	mean = np.mean(all_percentages_in)
	sem = stats.sem(all_percentages_in)
	n = len(all_percentages_in)
	results.append(["total stats", f"mean percentage in soma: {mean}", f"sem: {sem}", f"n: {n}"])

	df = pd.DataFrame(results, columns=["Neuron", "Total num SV", "Num in soma", "Num outside soma", "Percent in soma", "Percent outside soma", "Soma size"])
	df.to_excel("sv_soma_percentages.xlsx", index=False)
	print("export done")



















