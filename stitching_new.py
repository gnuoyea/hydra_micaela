import numpy as np
import h5py
import sys
import yaml

bbox = np.loadtxt('/data/projects/weilab/dataset/hydra/mask_mip1/bbox.txt').astype(int)
D0 = '/data/projects/weilab/dataset/hydra/results/'
D1 = '/data/rothmr/hydra/stitched/'

#dict by neuron, all zyx [neuron]: (corner), (other corner)
boxes = {}
for row in bbox:
	nid = row[0]
	corner1 = (row[1], row[3], row[5])
	corner2 = (row[2], row[4], row[6])
	boxes[nid] = ((corner1), (corner2))


cache={}
#use cache
def load_data(names_list):
	print("start loading data")

	for name in names_list:
		with h5py.File(f'{D0}neuron_{name}_30-32-32.h5', 'r') as f:
			cache[name] = f["main"][:]
			print("loaded ",name)


#range for a given axis, assume range is a1<a2, b1<b2
def ranges_overlap(a, b):
	a1, a2 = a
	b1, b2 = b

	if(b1<a2 and a1<b2):
		return True
	return False


def read_h5(filename, dataset=None):
    fid = h5py.File(filename, "r")
    if dataset is None:
        dataset = fid.keys() if sys.version[0] == "2" else list(fid)
    else:
        if not isinstance(dataset, list):
            dataset = list(dataset)

    out = [None] * len(dataset)
    for di, d in enumerate(dataset):
        out[di] = np.array(fid[d])

    return out[0] if len(out) == 1 else out

def merge_bbox(bbox_a, bbox_b):
    num_element = len(bbox_a) // 2 * 2
    out = bbox_a.copy()
    out[: num_element: 2] = np.minimum(bbox_a[: num_element: 2], 
                                       bbox_b[: num_element: 2])
    out[1: num_element: 2] = np.maximum(bbox_a[1: num_element: 2], 
                                        bbox_b[1: num_element: 2])
    if num_element != len(bbox_a): 
        out[-1] = bbox_a[-1] + bbox_b[-1]
    
    return out


#new method - small bbox is within larger bbox globally
#returns what part of the larger bbox to keep (local coords)
def find_crop(large, small): #takes in global coords
	z_min = small[0] - large[0] #lower min - higher min
	z_max = small[1] - large[0] + 1 #lower max - lower min

	y_min = small[2] - large[2]
	y_max = small[3] - large[2] + 1

	x_min = small[4] - large[4]
	x_max = small[5] - large[4] + 1

	bbox = [z_min, z_max, y_min, y_max, x_min, x_max]

	return bbox



def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data


def neuron_name_to_id(name):
	if isinstance(name, str):
		name = [name]
	return [neuron_dict[x] for x in name]  


#takes in either single id or list
def neuron_id_to_name(nid):
	reversed_dict = {v: k for k, v in neuron_dict.items()}
	if isinstance(nid, int):
		nid = [nid]
	return [reversed_dict[x] for x in nid]  


#needs to have input as one mask_file with size of nid1, then a list w/ [nid1, nid2]
#adds in data from nid2 to the mask file
def stitch_two(mask_file, nid):
	print("")
	print("nid: ", nid)
	names = neuron_id_to_name(nid)

	print("begin attempt to merge with ", names[1])

	bb1 = bbox[bbox[:,0]==nid[0], 1:][0]//[1,1,4,4,4,4]
	bb2 = bbox[bbox[:,0]==nid[1], 1:][0]//[1,1,4,4,4,4]
	
	bb_all = merge_bbox(bb1, bb2)
	bb_all_sz = (bb_all[1::2] - bb_all[::2]) + 1
	
	out = np.zeros(bb_all_sz, np.uint8)

	out[(bb1[0]-bb_all[0]) : (bb1[1]-bb_all[0]) + 1, \
	    (bb1[2]-bb_all[2]) : (bb1[3]-bb_all[2]) + 1, \
	    (bb1[4]-bb_all[4]) : (bb1[5]-bb_all[4]) + 1] = (mask_file)

	#use np.where here to avoid zeroing out stuff
	out_zero = (out==0) #places where the current mask is empty -> we can safely write data
	out[(bb2[0]-bb_all[0]) : (bb2[1]-bb_all[0]) + 1, \
	    (bb2[2]-bb_all[2]) : (bb2[3]-bb_all[2]) + 1, \
	    (bb2[4]-bb_all[4]) : (bb2[5]-bb_all[4]) + 1] = np.where(out_zero[(bb2[0]-bb_all[0]) : (bb2[1]-bb_all[0]) + 1, \
	    																	(bb2[2]-bb_all[2]) : (bb2[3]-bb_all[2]) + 1, \
	    																	(bb2[4]-bb_all[4]) : (bb2[5]-bb_all[4]) + 1], 
	    														nid[1] * cache[names[1]], 
	    														out[(bb2[0]-bb_all[0]) : (bb2[1]-bb_all[0]) + 1, \
	    															(bb2[2]-bb_all[2]) : (bb2[3]-bb_all[2]) + 1, \
	    															(bb2[4]-bb_all[4]) : (bb2[5]-bb_all[4]) + 1])

	#now crop
	cropped_bbox = find_crop(bb_all, bb1)
	cropped_out = out[cropped_bbox[0]:cropped_bbox[1], cropped_bbox[2]:cropped_bbox[3], cropped_bbox[4]:cropped_bbox[5]]

	print("successfully merged with ", names[1])

	return cropped_out


def stitch(name):
	neuron_id_1 = neuron_name_to_id(name)[0]

	to_stitch = [] #neuron IDs
	to_stitch_names = [] #names
	for neuron_id_2 in boxes:
		axes = (0,1,2) #xyz
		first = boxes[neuron_id_1]
		second = boxes[neuron_id_2]
	
		#if(neuron_id_1 != neuron_id_2):
		#check if all axis ranges overlap - includes the neuron itself
		if(ranges_overlap((first[0][0], first[1][0]), (second[0][0], second[1][0])) and 
			ranges_overlap((first[0][1], first[1][1]), (second[0][1], second[1][1])) and 
			ranges_overlap((first[0][2], first[1][2]), (second[0][2], second[1][2]))):

			#and check if it exists / is complete
			if(neuron_id_2 in neuron_dict.values()):
				to_stitch.append(neuron_id_2)
				to_stitch_names.append(neuron_id_to_name([neuron_id_2])[0])
			else:
				print("neuron ", neuron_id_2, " missing")

	#to_stitch should now contain IDs of: the current neuron plus all overlap box neurons
	print("BOX OVERLAPS: ", to_stitch)

	to_load = to_stitch_names.copy()

	load_data(to_load)
	print("done loading data")

	#create dataset
	stitched = np.zeros(cache[name].shape, dtype=cache[name].dtype)

	#stitch together in a loop using the stitch_two method
	for n in to_stitch:
		print("shape of stitched file: ", stitched.shape) #stays constant for each adjacent neuron
		print("seg ids in stitched file: ", np.unique(stitched))
		stitched = stitch_two(stitched, [neuron_id_1, n])

	print("final seg ids in stitched file: ", np.unique(stitched))

	fname = f'{D1}neuron_{name}_box_30-32-32.h5'
	with h5py.File(fname, 'w') as f:
		f.create_dataset("main", shape=stitched.shape, data=stitched)

	print("")
	print("saved stitched file as ", fname)


if __name__ == "__main__":
	neuron_dict = read_yml('/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt')

	#to_stitch = ['NET10', 'NET11', 'PN7', 'SHL18', 'SHL24', 'SHL28', 'RGC7', 'PN4', 'PN5', 'SHL26', 'SHL21', 'KM1', 'KR20', 'PN8', 'SHL29', 'SHL51', 'SHL52', 'SHL53', 'SHL54', 'KM2']

	to_stitch = neuron_dict.keys() #names of all neurons

	for n in to_stitch:
		print("")
		print("stitching ", n)
		stitch(n)

	










