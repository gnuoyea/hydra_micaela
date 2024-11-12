import edt
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
import os

#add global data cache or use memmap? for neurons & vesicles

#returns neurons (neuron + "adjacent" neurons) and corresponding vesicles - files are all in the +adj file shape
def load_stitched_data(nid):
	with h5py.File(f"neuron{nid:02}_friends.h5", 'r') as f:
		friends = f["main"][:]

	with h5py.File(f"neuron{nid:02}_ALL_vesicles.h5", 'r') as f:
		vesicles= f["main"][:]

	return neurons, vesicles


#calculate num of vesicles within any given mask 
#takes in mask and corresponding vesicles
def calculate_vesicles_within(mask, vesicles):
	#switch to a binary mask
	binary_mask = (mask>=1).astype(int)

	num_vesicles_within = 0

	labeled_vesicles = label(vesicles) #FOR STITCHED FILE, VESICLE LABELS CURRENTLY NOT UNIQUE (will fix in stitching.py)
	#TEMP FIX with no type counts: can change vesicles to binary file, then relabel everything
	vesicle_coords = np.column_stack(np.nonzero(labeled_vesicles))

	#optimized
	vesicle_coords_within_mask = vesicle_coords[binary_mask[tuple(vesicle_coords.T)] == 1]
	num_vesicles_within = len(vesicle_coords_within_mask)

	return num_vesicles_within


#currently unused - assuming neurons and vesicle files have same res
def convert_coords(original_coords, original_shape, original_res, target_shape, target_res):
    relative_coord = np.array(original_coords) / np.array(original_shape)
    target_coord = np.round(relative_coord * np.array(target_shape)).astype(int)
    return target_coord


def expand_mask(original_mask, threshold_nm, res): #threshold in physical units, res in xyz
	expanded_mask = np.copy(original_mask)

	dt = distance_transform_edt(original_mask, sampling=res) #in physical units
	doubled_perimeter = dt <= threshold_nm #all in nm
	expanded_mask[doubled_perimeter] = 1

	expanded_mask_binary = (expanded_mask>=1).astype(int)
	return expanded_mask_binary


def extract_perimeter(expanded_mask, original_mask):
	#expanded mask should already be binary
	#change original mask to binary
	original_mask_binary = (original_mask>=1).astype(int)
	original_mask_inverted = ~original_mask_binary

	#return everything in expanded but NOT in original
	perimeter_only_mask = np.copy(expanded_mask)
	perimeter_only_mask[perimeter_only_mask & original_mask_inverted] = 0

	return perimeter_only_mask


#return the intersection of all the given masks in the list, assume all same shape (add error checking later)
def mask_intersection(mask_list, shape):
	'''
	intersection = np.zeros(shape, dtype=bool)
	for mask in mask_list:
		binary_mask = (mask>=1).astype(int)
		intersection = intersection & binary_mask
	'''
	intersection = np.bitwise_and.reduce(mask_list)

	return intersection


#graph 1 column C
def vesicles_within_neuron(neurons, vesicles, nid):
    neuron_mask = (neurons == nid) #mask for single neuron, in +adj mask dims
    return calculate_vesicles_within(neuron_mask, vesicles)


 #graph 1 column E (and all of graph 2)
 #total within the perimeter of the given nid only
def total_within_perimeter(neurons, vesicles, nid, threshold_nm, res):
	#get singular neuron (for nid)
 	singular_neuron = (neurons == nid)

 	expanded_mask = expand_mask(singular_neuron, threshold_nm, res) #RES IN XYZ
 	perimeter_only = extract_perimeter(expanded_mask, singular_neuron)

 	return calculate_vesicles_within(perimeter_only, vesicles)


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

	#get current perimeter mask for current neuron and append to the all_masks list
	current_neuron = (neurons == nid)
	expanded_mask = expand_mask(current_neuron, threshold_nm, res) #current neuron
 	perimeter_only = extract_perimeter(expanded_mask, original_mask)
	all_masks.append(perimeter_only)

	for adjacent_neuron in adjacent_neurons:
		adjacent_mask = (neurons == adjacent_neuron)
		expanded_adjacent = expand_mask(adjacent_mask, threshold_nm, res)
		all_masks.append(adjacent_mask)

	overlaps_mask = mask_intersection(all_masks, neurons.shape)
	overlaps_count = calculate_vesicles_within(overlaps_mask, vesicles)
	overlaps_volume = calculate_volume_nm(overlaps_mask, res)

	return overlaps_count, overlaps_volume


if __name__ == "__main__":
	res_xyz = (8,8,30)

	nid = 38
	neurons, vesicles = load_stitched_data(nid)
	print(f"within neuron {nid}: ", vesicles_within_neuron(neurons, vesicles, nid))
	print(f"within perimeter for neuron {nid}: ", total_within_perimeter(neurons, vesicles, nid, threshold_nm = 1000, res = res_xyz)) #1 micron
	overlaps_count, overlaps_volume = near_another_neuron(neurons, vesicles, nid, threshold_nm = 1000, res = res_xyz)
	print(f"within perimeter for neuron {nid} & near another neuron: ", overlaps_count) #1 micron
	print(f"volume (nm^3) of perimeter overlaps for neuron {nid} & adjacent expansions: ", overlaps_volume) #1 micron







