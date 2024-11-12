import numpy as np
import h5py

'''
TO-DO: (during stitching process)
NEED TO MAP ALL THE VESICLE COORDS INTO NEW COORDS FOR STITCHED FILES - create a dict linking the new unique id to a tuple
'''

#add a data loader method

#current spreadsheet as of 11-09; fill in rest later
#xyz
boxes = {38: ((40872, 64528, 625), (116504, 77180, 1786)), 
		41: ((111960, 68964, 577), (117296, 76200, 1065)), 
		37: ((110244, 71768, 403), (114892, 76392, 1356)),
		15: ((99100, 64895, 412), (104233, 73266, 1084)), 
		39: ((113086, 64312, 640), (124908, 73508, 1234)),
		16: ((45636, 61172, 360), (89081, 80289, 941)),
		36: ((115992, 63914, 503), (119820, 73000, 905))}

def stitch(neuron_id_1):
	to_stitch = [] #neuron IDs
	for neuron_id_2 in boxes:
		axes = (0,1,2) #xyz
		first = boxes[neuron_id_1]
		second = boxes[neuron_id_2]
	
		
		#check if all axis ranges overlap
		if(ranges_overlap((first[0][0], first[1][0]), (second[0][0], second[1][0])) and 
			ranges_overlap((first[0][1], first[1][1]), (second[0][1], second[1][1])) and 
			ranges_overlap((first[0][2], first[1][2]), (second[0][2], second[1][2]))):

			to_stitch.append(neuron_id_2)

	#to_stitch should now contain IDs of: the current neuron plus all overlap box neurons
	return to_stitch

	#stitch together and return h5
	#find extreme values in to_stitch to create blank h5 file
	
	min_x = min(boxes[nid][0][0] for nid in to_stitch)
	min_y = min(boxes[nid][0][1] for nid in to_stitch)
	min_z = min(boxes[nid][0][2] for nid in to_stitch)

	max_x = max(boxes[nid][1][0] for nid in to_stitch)
	max_y = max(boxes[nid][1][1] for nid in to_stitch)
	max_z = max(boxes[nid][1][2] for nid in to_stitch)

	#create friends blank h5 file
	friends_filename = f"neuron{neuron_id_1:02}_friends.h5" # FILE 1
	with h5py.File(friends_filename, 'w') as f:
		shape = (max_x-min_x, max_y-min_y, max_z-min_z)
		friends_data = f.create_dataset("main", shape=shape, dtype=np.uint8)
		friends_data[:] = np.zeros(shape, dtype=uint8)

		#fill data with each neuron mask, coords adjusted
		for nid in to_stitch:
			with h5py.File(f"neuron{nid:02}_mask.h5", 'r') as f:
				loaded_neuron = f["main"][:] #zyx
			#translated to zyx
			friends_data[boxes[nid][0][2]-min_z:boxes[nid][1][2]-min_z, boxes[nid][0][1]-min_y:boxes[nid][1][1]-min_y, boxes[nid][0][0]-min_x:boxes[nid][1][0]-min_x] =
			loaded_neuron[:,:,:]
	#neuron friends data is all loaded

	#create vesicles blank h5 file (does this need to be scaled / change res?)
	vesicles_filename = f"neuron{neuron_id_1:02}_ALL_vesicles.h5" #vesicles for all the adjacent neurons too # FILE 2

	with h5py.File(vesicles_filename, 'w') as f:
		shape = (max_x-min_x, max_y-min_y, max_z-min_z)
		vesicles_data = f.create_dataset("main", shape=shape, dtype=np.uint8)
		vesicles_data[:] = np.zeros(shape, dtype=uint8)

		#fill data with each vesicle file, coords adjusted
		for nid in to_stitch:
			with h5py.File(f"neuron{nid:02}_vesicles.h5", 'r') as f:
				loaded_vesicles = f["main"][:] #zyx
			#translated to zyx
			vesicles_data[boxes[nid][0][2]-min_z:boxes[nid][1][2]-min_z, boxes[nid][0][1]-min_y:boxes[nid][1][1]-min_y, boxes[nid][0][0]-min_x:boxes[nid][1][0]-min_x] =
			loaded_vesicles[:,:,:]
	#neuron ALL vesicles data is all loaded


#range for a given axis, assume range is a1<a2, b1<b2
def ranges_overlap(a, b):
	a1, a2 = a
	b1, b2 = b

	if(b1<a2 and a1<b2):
		return True
	return False

if __name__ == "__main__":
	stitch(38) #generate files






