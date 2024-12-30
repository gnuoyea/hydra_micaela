import numpy as np
import h5py
from scipy.ndimage import distance_transform_edt as edt
import numpy as np
import yaml

cache = {}
D1 = '/data/rothmr/hydra/stitched/'
D3 = '/data/rothmr/hydra/sa/' #exporting for visualization

def load_data(name):
	with h5py.File(f"{D1}neuron_{name}_box_30-32-32.h5", 'r') as f:
		cache["box"] = f["main"][:]

def read_yml(filename):
	with open(filename, 'r') as file:
		data = yaml.safe_load(file)
	return data

def neuron_name_to_id(name):
	if isinstance(name, str):
		name = [name]
	return [neuron_dict[x] for x in name] 

def expand_mask(original_mask, threshold_nm, res): #threshold in physical units, res in xyz
	expanded_mask = np.copy(original_mask)

	print("begin distance transform for expansion")
	dt = edt.edt(1 - original_mask, anisotropy=(res[2],res[1],res[0])) #needs to be in xyz by default for edt
	print("end distance transform for expansion")
	doubled_perimeter = dt <= threshold_nm #all in nm
	expanded_mask[doubled_perimeter] = 1

	expanded_mask_binary = (expanded_mask>=1).astype(int)
	return expanded_mask_binary


def surface_area(mask, name, nid):
	other_neurons = ((mask!=nid) & (mask!=0)).astype(np.uint16)
	neuron = (mask==nid).astype(np.uint16)

	print("start edt for border")
	neuron_edt = edt(neuron)
	print("end edt for border")
	neuron_border = ((neuron_edt>0) * (neuron_edt<=1)).astype(np.uint16)

	border_voxels = int(neuron_border.sum())
	print(f'{name} total SA (nm^2): {border_voxels*32**2}') #output 1

	print("start edt for expansion")
	other_neurons_edt = edt(1-other_neurons)
	print("end edt for expansion")
	expansion = ((other_neurons_edt>0) * (other_neurons_edt<=3)).astype(np.uint16) #3 voxels expansion

	intersection = np.bitwise_and(neuron_border, expansion)
	intersection_voxels = int(intersection.sum())

	print(f'{name} near neuron SA (nm^2): {intersection_voxels*32**2}') #output 2


	'''
	#optional for visualization: checking that the expansion is enough - doesn't affect actual code
	fname = f'{D3}surfaces_{name}.h5'
	with h5py.File(fname, 'w') as f:
		f.create_dataset("main", data=intersection, shape = intersection.shape, dtype=intersection.dtype) #or dtype=np.uint8 ?
	print(f"successfully exported surface file as {fname}")

	fname = f'{D3}border_{name}.h5'
	with h5py.File(fname, 'w') as f:
		f.create_dataset("main", data=neuron_border, shape = neuron_border.shape, dtype=neuron_border.dtype) #or dtype=np.uint8 ?
	print(f"successfully border surface file as {fname}")

	fname = f'{D3}expansion_{name}.h5'
	with h5py.File(fname, 'w') as f:
		f.create_dataset("main", data=expansion, shape = expansion.shape, dtype=expansion.dtype) #or dtype=np.uint8 ?
	print(f"successfully expansion surface file as {fname}")
	'''

#no padding, loads chunks on their own
def load_chunks(mask, chunk_num, num_chunks, key):
	shape = mask.shape
	dtype = mask.dtype

	#calculate chunk_length (last chunk might be this plus some remainder if this doesn't divide evenly)
	chunk_length = (shape[1])//num_chunks #integer division, dividing up y axis length

	if(chunk_num!=num_chunks-1):
		output = mask[:, chunk_num*chunk_length:(chunk_num+1)*chunk_length, :]
		cache[key] = output

	else: #case of the last chunk
		output = mask[:, chunk_num*chunk_length:, :] #go to end of the file - last chunk includes any leftover stuff
		cache[key] = output

	print(f"done loading {key} for chunk #{chunk_num+1}") #bc zero indexing


#process in chunks due to edt taking to long
def surface_area_chunks(mask, name, nid, num_chunks):
	neuron = (mask==nid).astype(np.uint16)
	other_neurons = ((mask!=nid) & (mask!=0)).astype(np.uint16)

	current_chunk = 0
	total_surface_area = 0
	intersection_surface_area = 0

	while(current_chunk!=num_chunks): #this loop runs [num_chunks] times
		print(f"processing chunk {current_chunk+1}")
		load_chunks(neuron, current_chunk, num_chunks, "neuron")
		#cache["neuron"] now holds the current chunk of neuron mask
		load_chunks(other_neurons, current_chunk, num_chunks, "other_neurons")
		#cache["other_neurons"] now holds the current chunk of other_neurons mask
		#these files are cut off on each side (no padding) -> expansions not affected

		print("start edt for border")
		neuron_edt = edt(cache["neuron"])
		print("end edt for border")
		neuron_border = ((neuron_edt>0) * (neuron_edt<=1)).astype(np.uint16)

		border_voxels = int(neuron_border.sum())
		total_surface_area += (border_voxels*32**2) #output 1


		print("start edt for expansion")
		other_neurons_edt = edt(1-cache["other_neurons"])
		print("end edt for expansion")
		expansion = ((other_neurons_edt>0) * (other_neurons_edt<=3)).astype(np.uint16) #3 voxels expansion

		intersection = np.bitwise_and(neuron_border, expansion)
		intersection_voxels = int(intersection.sum())
		intersection_surface_area += (intersection_voxels*32**2)

		current_chunk += 1

	print(f'{name} total SA (nm^2): {total_surface_area}') #output 1
	print(f'{name} near neuron SA (nm^2): {intersection_surface_area}\n') #output 2



if __name__ == "__main__":
	neuron_dict = read_yml('/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt')

	#to_calculate = ['SHL17']
	to_calculate = neuron_dict.keys()

	chunking = True
	num_chunks = 4
	
	for name in to_calculate:
		print(f"calculating surface area for {name}")
		nid = neuron_name_to_id(name)[0]
		load_data(name)

		if chunking:
			surface_area_chunks(cache["box"], name, nid, num_chunks)
		else:
			surface_area(cache["box"], name, nid)




























