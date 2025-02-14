import numpy as np
import h5py
from scipy.ndimage import center_of_mass
import scipy.spatial as spatial
import ast
import math

#goal 1: generate metadata for each vesicle based on COM for stats exporting, types visualization etc
#goal 2: export a mapping between each vesicle COM (and its radius) and the density in its respective spatial neighborhood for 3d reconstruction
#note - everything is DS by 2!
#citation for kdtree stuff: https://stackoverflow.com/questions/14070565/calculating-point-density-using-python

D0 = '/data/projects/weilab/dataset/hydra/results/'
D_metadata = '/data/rothmr/hydra/metadata/'
D_coms = '/data/rothmr/hydra/coms/'

cache = {}
pointcloud = {}

#loads with everything in 30-16-16 b/c just extracting COMs
def load_data(name, which):
	if(which=="lv"):
		if(name=="SHL17"): #fixing too large bbox
			with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f:
				cache["lv"] = np.array(f["main"][:, 4000:, :2400][:, ::8, ::8]) #30-64-64
		else:
			with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f:
				cache["lv"] = np.array(f["main"][:][:, ::8, ::8]) #30-64-64
	if(which=="sv"):
		if(name=="SHL17"): #fixing too large bbox
			with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f:
				cache["sv"] = np.array(f["main"][:, 4000:, :2400][:, ::8, ::8]) #30-64-64
		else:
			with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f:
				cache["sv"] = np.array(f["main"][:][:, ::8, ::8]) #30-64-64

def read_pointcloud_from_file(filename):
	pointcloud = {}
	with open(filename, 'r') as f:
		for line in f:
			#print(f"line from file: {line.strip()}") #testing
			try:
				com_str, label = line.strip().split(": ")
				com_str = com_str.replace("np.float64", "").replace("(", "").replace(")", "")
				com = tuple(map(float, com_str.split(',')))
				label = float(label)
				pointcloud[com] = label
			except Exception as e:
				print(f"Error processing line: {line.strip()}")
				print(f"Error: {e}")
	return pointcloud

#note the files are in 30x16x16 at this point
#initiate all metadata rahhhh
def init_pointcloud(name):
	#load lv
	load_data(name, "lv") #only load lv
	print(f"done loading lv for {name}")

	#extract the COMs of each vesicle for LV
	vesicles = cache["lv"].astype(np.float16)
	unique_labels = np.unique(vesicles)
	num_labels = len(unique_labels)

	#for LV
	for label in unique_labels:
		if(label!=0): #label is local and specific to this neuron lv
			binary_mask = (vesicles == label) #extract the one vesicle
			center = center_of_mass(binary_mask)
			which = "lv"
			volume_nm = np.sum(binary_mask) * (30*64*64)
			#find radius by collapsing into the z axis then scale to get to nm
			circle = np.sum(binary_mask, axis=0)
			circle_area = np.sum(circle) #this is in in [30,16,16], but only the x and y coords
			circle_area_nm = circle_area * 64 * 64
			radius_nm = np.sqrt(circle_area_nm/np.pi) #sqrt(A/pi)
			print("COM: ", center)
			attributes = (which, label, volume_nm, radius_nm)
			pointcloud[center] = attributes
			print(f"attributes: {attributes}") #for initial checking
	del cache["lv"]

	#just in case
	with open(f"{D_coms}{name}_lv_com_mapping.txt", "w") as f:
		for center, attributes in pointcloud.items():
			f.write(f"{center}: {attributes}\n")
	print(f"initial lv mapping saved to {name}_lv_com_mapping.txt")

	#do the same with SV
	load_data(name, "sv")
	print(f"done loading sv for {name}")

	vesicles = cache["sv"].astype(np.float16)
	unique_labels = np.unique(vesicles)
	num_labels = len(unique_labels)

	for label in unique_labels:
		if(label!=0): #label is local and specific to this neuron sv
			binary_mask = (vesicles == label) #extract the one vesicle
			center = center_of_mass(binary_mask)
			which = "sv"
			volume_nm = np.sum(binary_mask) * (30*64*64)
			#find radius by collapsing into the z axis then scale to get to nm
			circle = np.sum(binary_mask, axis=0)
			circle_area = np.sum(circle) #this is in in [30,16,16], but only the x and y coords
			circle_area_nm = circle_area * 64 * 64
			radius_nm = np.sqrt(circle_area_nm/np.pi) #sqrt(A/pi)
			print("COM: ", center)
			attributes = (which, label, volume_nm, radius_nm)
			pointcloud[center] = attributes
			print(f"attributes: {attributes}") #for initial checking
	del cache["sv"]

	#just in case
	with open(f"{D_coms}{name}_sv_com_mapping.txt", "w") as f:
		for center, attributes in pointcloud.items():
			f.write(f"{center}: {attributes}\n")
	print(f"initial sv mapping saved to {name}_sv_com_mapping.txt")


#due to type casting and floating point 
def are_coords_equal(c1, c2, tolerance=1e-6):
	return all(abs(a - b) < tolerance for a, b in zip(c1, c2))

'''
each point has 3 features, each level of the tree alternates between sorting parent/child based on each dim, then walk down the tree
to find a candidate for nearest neighbor. then check if radius is greater than distance to other side of the splitting plane.
'''
def kd_tree(coords, name):
	#convert everything into tuples instead of np arrays - for hashable type
	coords = [tuple(float(x) for x in coord) if isinstance(coord, np.ndarray) else coord for coord in coords]
	print("types conversion done for coords list")

	tree = spatial.KDTree(coords)
	radius = 500 #in nm, can adjust this as needed
	neighbors = tree.query_ball_tree(tree, radius) #this is a list of coords
	frequency = np.array([len(i) for i in neighbors])
	density = frequency/radius**2
	print("checkpoint 1")
	density_coords_mapping = {(coords[i]): density[i] for i in range(len(density))} #maps coords to densities
	print("length of dict: ", len(density_coords_mapping))
	#normalize
	density_values = np.array(list(density_coords_mapping.values()))
	min_density = np.min(density_values)
	max_density = np.max(density_values)
	normalized_densities = (density_values-min_density) / (max_density-min_density)
	normalized_density_coords_mapping = {coord: normalized_densities[i] for i, (coord, normalized_density) in enumerate(zip(density_coords_mapping.keys(), normalized_densities))}

	#now do the same but ONLY with the lv coords
	#extract lv coords from coords
	coords_arr = np.array(list(pointcloud.keys()))
	attributes_arr = np.array(list(pointcloud.values()), dtype=object)
	lv_mask = attributes_arr[:, 0] == 'lv'
	lv_coords = coords_arr[lv_mask]

	#most of variables fine to overwrite only make new for the kdtree and final mapping
	lv_tree = spatial.KDTree(coords)
	radius = 500 #in nm, can adjust this as needed
	neighbors = lv_tree.query_ball_tree(lv_tree, radius) #this is a list of coords
	frequency = np.array([len(i) for i in neighbors])
	density = frequency/radius**2
	print("checkpoint 1")
	density_coords_mapping = {(coords[i]): density[i] for i in range(len(density))} #maps coords to densities
	print("length of dict: ", len(density_coords_mapping))
	#normalize
	density_values = np.array(list(density_coords_mapping.values()))
	min_density = np.min(density_values)
	max_density = np.max(density_values)
	normalized_densities = (density_values-min_density) / (max_density-min_density)
	lv_normalized_density_coords_mapping = {coord: normalized_densities[i] for i, (coord, normalized_density) in enumerate(zip(density_coords_mapping.keys(), normalized_densities))}


	with open(f"{name}_metadata.txt", "w") as f:
		for coord in normalized_density_coords_mapping.keys():
			norm_dens = normalized_density_coords_mapping[coord]
			lv_norm_dens = lv_normalized_density_coords_mapping[coord]
			attributes = pointcloud[coord]
			print(f"{coord}: {attributes.append(norm_dens, lv_norm_dens)} \n") #check format is ok
			f.write(f"{coord}: {attributes.append(norm_dens, lv_norm_dens)} \n")

	print(f"metadata saved to {D_metadata}{name}_metadata.txt")
	#saves the metadata with EVERYTHING in nm


if __name__ == "__main__":
	name = "SHL28"

	generate_pointcloud = True #this is the time consuming task so can do separately

	if(generate_pointcloud):
		init_pointcloud(name)
		print("init pointcloud done")

	#once pointcloud already initiated
	print("load pointcloud")
	pointcloud = read_pointcloud_from_file(f"{name}_com_mapping.txt")
	print("done loading pointcloud")
	print("length of original pointcloud: ", len(pointcloud))

	res = [30, 64, 64]	
	#create np array of only the COM coords
	com_coords_voxel = np.array(list(pointcloud.keys()))
	com_coords_nm = com_coords_voxel * res #convert to physical units

	pointcloud = {tuple(com_coords_nm[i]): pointcloud[key] for i, key in enumerate(pointcloud)} #saves to cache in nm

	kd_tree(com_coords_nm, name) #uses pointcloud from cache
	#saves the metadata with EVERYTHING in nm

	





