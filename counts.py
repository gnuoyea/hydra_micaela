import edt
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
import os

cache = {}

#all files should be in the shape of the neuron mask file for the neuron name eg. "KR5"
def load_data(name):
	with h5py.File(f"neuron_{name}_box_30-8-8.h5", 'r') as f: #stitched high res data mask, contains everything w segids
		cache["box"] = f["main"][:]

	with h5py.File(f"lv_{name}_30-8-8.h5", 'r') as f: #high res large vesicles data for [name]
		cache["lv"] = f["main"][:]

	with h5py.File(f"sv_{name}_30-8-8.h5", 'r') as f: #high res small vesicles data for [name]
		cache["sv"] = f["main"][:]


#calculate num of vesicles within any given mask; takes in mask and corresponding vesicles file
def calculate_vesicles_within(mask, vesicles):
	#switch to a binary mask
	binary_mask = (mask>=1).astype(int)

	num_vesicles_within = 0

	#labeled_vesicles = label(vesicles)
	labeled_vesicles = vesicles #no need to relabel
	vesicle_coords = np.column_stack(np.nonzero(labeled_vesicles))

	unique_labels = np.unique(labeled_vesicles)
	num_labels = len(unique_labels) - 1 #minus the bg label

	vesicles_within_mask = np.zeros(num_labels, dtype=bool)
	vesicle_coords = np.column_stack(np.nonzero(labeled_vesicles))
	mask_values = binary_mask[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]]
	vesicles_within_mask = np.unique(labeled_vesicles[vesicle_coords[:, 0], vesicle_coords[:, 1], vesicle_coords[:, 2]][mask_values == 1])
	num_vesicles_within = len(vesicles_within_mask)

	return num_vesicles_within


#currently unused since neurons and vesicle files now have same res
def convert_coords(original_coords, original_shape, original_res, target_shape, target_res):
    relative_coord = np.array(original_coords) / np.array(original_shape)
    target_coord = np.round(relative_coord * np.array(target_shape)).astype(int)
    return target_coord


def expand_mask(original_mask, threshold_nm, res): #threshold in physical units, res in zyx
	expanded_mask = np.copy(original_mask)

	dt = edt.edt(1 - original_mask, anisotropy=(res[2],res[1],res[0])) #needs to be in xyz by default for edt
	perimeter = dt <= threshold_nm #all in nm
	expanded_mask[perimeter] = 1

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


#graph 1 column C
def vesicles_within_neuron(neurons, vesicles, nid):
    neuron_mask = (neurons == nid) #mask for single neuron, in +adj mask dims
    return calculate_vesicles_within(neuron_mask, vesicles)


#graphs 1, 2, 3
#calculate the volume given a boolean mask & convert to nm^3 units
def calculate_volume_nm(mask, res):
	binary_mask = (mask>=1).astype(int)
	volume_voxels = np.sum(binary_mask)
	volume_nm = volume_voxels * (res[0]*res[1]*res[2])
	return volume_nm


#return the total overlaps count & overlaps mask volume - later use overlaps mask vol for rand region
#threshold in nm defines what "near" means
def near_another_neuron(neurons, vesicles, nid, threshold_nm, res):
    #list of everything in the adjacency chunk besides current nid
    adjacent_neurons = []

	#construct adjacent_neurons
    adjacent_neurons = [label for label in np.unique(neurons) if label!=nid]

    all_masks = [] #to calculate the intersection of later

	#get mask for current neuron and append to the all_masks list
    current_neuron = (neurons == nid)
    all_masks.append(current_neuron)

    for adjacent_neuron in adjacent_neurons:
        adjacent_mask = (neurons == adjacent_neuron)
        expanded_adjacent = expand_mask(adjacent_mask, threshold_nm, res)
        all_masks.append(adjacent_mask)

    overlaps_mask = mask_intersection(all_masks, neurons.shape)
    overlaps_count = calculate_vesicles_within(overlaps_mask, vesicles)
    overlaps_volume = calculate_volume_nm(overlaps_mask, res)

    return overlaps_count, overlaps_volume


if __name__ == "__main__":
	res = (30,8,8)
	name = "KR5"
	nid = 38
	threshold = 1000 #in nm, adjust as needed
	docked_threshold = 250 #by default
	load_data(name)

	#example usage:

	print(f"LV within neuron {name}: ", vesicles_within_neuron(cache["box"], cache["lv"], nid))
	overlaps_count, overlaps_volume = near_another_neuron(cache["box"], cache["lv"], nid, threshold_nm = threshold, res = res)
	print(f"LV within neuron {name} & within {threshold}nm of another neuron: ", overlaps_count)
	print(f"Volume (in nm^3) of overlaps for neuron {name} and {threshold}nm: ", overlaps_volume)
	num_docked, vol_eroded = docked_vesicles(((cache["box"])==nid), cache["lv"], docked_threshold, res)
	print(f"LV docked vesicles for neuron {name} and threshold {docked_threshold}nm: ", num_docked)
	print(f"Vol of eroded region for docked vesicles: ", vol_eroded)







