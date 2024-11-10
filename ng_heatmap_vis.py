import h5py
import os,sys
from em_util.em_util.io.io import *
from em_util.em_util.ng import *
import numpy as np
from scipy.ndimage import zoom

import neuroglancer
import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

ip = 'localhost' #or public ip for sharable display
port = find_free_port()
neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
viewer=neuroglancer.Viewer()


res = [8, 8, 30] #vesicle predictions 
res_ds = [16, 16, 60] #downsampled for density map
resM = [128, 128, 120] #neuron mask

#for density map - process 4d input instead of 3d
def ng_layer_edited(data, res, oo=[0, 0, 0], tt="image"):
    if data.ndim == 4:
        #shape is (d,h,w,c)
        data = data.transpose(2, 1, 0, 3)  #now (h,w,d,c)
        data = data.mean(axis=-1)  #average color channels -> (d,h,w)

        #clip and change to uint8
        data = np.clip(data, 0, 255).astype(np.uint8)

    dim = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=res)
    return neuroglancer.LocalVolume(data, volume_type=tt, dimensions=dim, voxel_offset=oo)


def visualize():
    full_mask = read_vol(f"cropped_mask_7-13.h5") #already cropped and scaled from prev script
    vesicles = read_vol("ngs_colored_density_map_7-13.h5")

    with viewer.txn() as s:
        s.layers.append(name='neuron_mask',layer=ng_layer(full_mask,res=resM))

        print("map shape: ", vesicles.shape) #(94, 2048, 3, 4)
        vesicles_transpose = vesicles.transpose(1,2,3,0) #3,0,1,2 #2,1,0,3
        print("Shape of vesicles after transpose:", vesicles_transpose.shape)

        s.layers.append(name='density_map',layer=ng_layer_edited(vesicles,res=res_test, tt='image')

visualize()


print(viewer)
