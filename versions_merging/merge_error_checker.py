import numpy as np
import h5py
import re
import argparse
import os

#generate old and new metadata separately, then combine
#checking for errors


names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
			"RGC2", "KM4", "SHL17", "NET12", "NET10", "NET11", "PN7", "SHL18", 
			"SHL24", "SHL28", "RGC7"]


#for the old data
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


def add_extra(name): 
	#add error checking - if vesicle IDs are overwritten

	old_types_path = f"types_lists/{name}_types.txt" #in id
	extra_types_path = f"extra_lv_types/extra-{name}txt/" #folder names (stripped)

	combined_types_path = f"combined_lv_types/{name}_types.txt" #init everything new

	old_stuff_dict = read_txt_to_dict(old_types_path)

	overlaps = []
	for item in os.listdir(extra_types_path):
		if(item != ".DS_Store"):
			current_file_path = os.path.join(extra_types_path, item)
			print(current_file_path)
			extra_stuff_dict = read_txt_to_dict(current_file_path)

			#check if any IDs overlap between old and extra
			added_overlaps = [key for key,value in extra_stuff_dict.items() if (key
								in old_stuff_dict.keys() and value!=old_stuff_dict[key])]
							#"value" is from extra(new), check if new value is diff from old

			overlaps.extend(added_overlaps)

	print(f"vesicle ID conflicts for {name} ", overlaps)

	return overlaps



if __name__ == "__main__":
	names = names_20


	#do once
	#rename file paths - strip spaces
	path="extra_lv_types"
	for item in os.listdir(path):
		old_name = os.path.join(path,item)
		if os.path.isdir(old_name):
			new_name = old_name.replace(" ", "")
			os.rename(old_name, new_name)


	output_file = f"mapping_conflicts.txt"

	for name in names: 
		overlaps = add_extra(name)
		with open(output_file, "a") as f:
			f.write(f"{name}: {overlaps} \n \n")


