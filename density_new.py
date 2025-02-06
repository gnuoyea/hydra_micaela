import numpy as np
import h5py
from scipy.ndimage import center_of_mass
import scipy.spatial as spatial
import ast
import math

#not yet integrated lv and sv, this is only for lv - also ds by 2

#goal: export a mapping between each vesicle and the density in its respective spatial neighborhood for later 3d mesh visualization
#citation for kdtree stuff: https://stackoverflow.com/questions/14070565/calculating-point-density-using-python

D0 = '/data/projects/weilab/dataset/hydra/results/'

cache = {}
pointcloud = {} #COM -> label (in voxel coords, for exporting)

#downsample bc of the full ng vis
#might need to implement chunks later
def load_data(name):
    if(name=="SHL17"): #fixing too large bbox
        with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f:
            cache["lv"] = np.array(f["main"][:, 4000:, :2400][:, ::2, ::2]) #30-16-16
        '''
        with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f:
            cache["sv"] = np.array(f["main"][:, 4000:, :2400][:, ::2, ::2]) #30-16-16
        '''
    else:
        with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f:
            cache["lv"] = np.array(f["main"][:][:, ::2, ::2]) #30-16-16
        '''
        with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f:
            cache["sv"] = np.array(f["main"][:][:, ::2, ::2]) #30-16-16
        '''

'''
#create combined file, relabel vesicles - do later
#(only relabel SV? by adding offset based on the highest LV label)
def combine_vesicles():
	#intialize combined file, relabel actual sv in h5, clear cache, save new file in cache

	#save mapping dict for global-> local SV in the cache
'''

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

#initiates to voxel coords
def init_pointcloud():
	#extract the COMs of each vesicle
	vesicles = cache["lv"].astype(np.float16)
	unique_labels = np.unique(vesicles)
	num_labels = len(unique_labels)

	for label in unique_labels:
		if(label!=0):
			binary_mask = (vesicles == label) #extract the one vesicle
			center = center_of_mass(binary_mask)
			print("COM: ", center)
			pointcloud[center] = label

	with open(f"{name}_com_mapping.txt", "w") as f:
		for center, label in pointcloud.items():
			f.write(f"{center}: {label}\n")

	print(f"mapping saved to {name}_com_mapping.txt")


	'''
	for label in unique_labels:
		indices = np.array(np.where(vesicles))
		binary_mask = vesicles==label
		weighted_sum = np.sum(indices.T * binary_mask[indices[:, 0], indices[:, 1], indices[:, 2]], axis=1)
		mass = np.sum(binary_mask)
		com = weighted_sum/mass 
		print(com)
		pointcloud[label] = com
	'''

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
	radius = 500 #adjust this as needed

	neighbors = tree.query_ball_tree(tree, radius) #this is a list of coords
	frequency = np.array([len(i) for i in neighbors])
	density = frequency/radius**2

	print("checkpoint 1")

	density_coords_mapping = {(pointcloud[coords[i]]): density[i] for i in range(len(density))}
	print("length of dict: ", len(density_coords_mapping))

	#normalize?
	density_values = np.array(list(density_coords_mapping.values()))
	min_density = np.min(density_values)
	max_density = np.max(density_values)
	normalized_densities = (density_values-min_density) / (max_density-min_density)
	normalized_density_coords_mapping = {int(segid): normalized_densities[i] for i, (segid, normalized_density) in enumerate(zip(density_coords_mapping.keys(), normalized_densities))}
	print("mapping done")

	with open(f"{name}_density_mapping.txt", "w") as f:
		for segid, dens in normalized_density_coords_mapping.items():
			f.write(f"{segid}: {dens}\n")

	print(f"mapping saved to {name}_density_mapping.txt")


if __name__ == "__main__":
	name = "KR6"

	generate_pointcloud = True #this is the time consuming task do separately

	if(generate_pointcloud):
		load_data(name) #only load lv
		print(f"done loading data for {name}")
		init_pointcloud()
		print("init pointcloud done")

	#once pointcloud already initiated
	print("load pointcloud")
	pointcloud = read_pointcloud_from_file(f"{name}_com_mapping.txt")
	print("done loading pointcloud")
	print("length of original pointcloud: ", len(pointcloud))

	res = [30, 16, 16]	
	#create np array of only the COM coords
	com_coords_voxel = np.array(list(pointcloud.keys()))
	com_coords_nm = com_coords_voxel * res #convert to physical units

	pointcloud = {tuple(com_coords_nm[i]): pointcloud[key] for i, key in enumerate(pointcloud)}

	kd_tree(com_coords_nm, name)

	





