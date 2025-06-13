import numpy as np
import h5py
import neuroglancer
import imageio
import socket
from contextlib import closing
import yaml
import os
import gc
from PIL import Image
import matplotlib.pyplot as plt


bbox = np.loadtxt('/data/projects/weilab/dataset/hydra/mask_mip1/bbox.txt').astype(int)

cache = {}

#only store one at a time
def load_data(name):
    with h5py.File(f"color_test.h5", "r") as f:
        cache["vesicles"] = np.array(f["main"][:]) #30-16-16
    with h5py.File(f"/data/projects/weilab/dataset/hydra/results/neuron_{name}_30-32-32.h5") as f:
        cache["mask"] = np.array(f["main"][:]) #30-32-32

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

ip = 'localhost' #or public IP of the machine for sharable display
port = find_free_port() #change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
viewer=neuroglancer.Viewer()


# SNEMI (# 3d vol dim: z,y,x)
D0='./'
res1 = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units=['nm', 'nm', 'nm'],
        scales = [30,64,64])

res2 = neuroglancer.CoordinateSpace(
        names=['z','y','x'],
        units=['nm', 'nm', 'nm'],
        scales=[30,16,16])

res3 = neuroglancer.CoordinateSpace(
        names=['z','y','x'],
        units=['nm', 'nm', 'nm'],
        scales=[30,32,32])


def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
    return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)        

def neuron_name_to_id(name):
    if isinstance(name, str):
        name = [name]
    return [neuron_dict[x] for x in name]

def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

#find the min bbox coord (bottom left corner) for this neuron, return as list of 3 coords
def get_offset(name): #returns in 30-8-8
    nid = neuron_name_to_id(name)[0]
    bb = bbox[bbox[:,0]==nid, 1:][0]
    output = [bb[0], bb[2], bb[4]]
    
    return output


def screenshot(path='temp.png', save=True, show=True, size=[4096, 4096]):
    ss = viewer.screenshot(size=size).screenshot.image_pixels
    if save:
        Image.fromarray(ss).save(path)
    if show:
        plt.imshow(ss)
        plt.show()


with viewer.txn() as s:

    ##COLOR CODED VIS
    names = ['KR6'] #test

    neuron_dict = read_yml('/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt')

    for name in names:
        path = f"color_test.h5"
        if(os.path.exists(path)):
            load_data(name) #cache["vesicles"] is updated to the current neuron's vesicle data
            print(f"done loading data for {name}")

            vesicles = cache["vesicles"]
            mask = cache["mask"]

            s.layers.append(name=f'{name}_mask',layer=ngLayer(mask,res=res3, tt='segmentation')) #30-32-32
            s.layers.append(name=f'{name}_yellow',layer=ngLayer((vesicles==1).astype('uint16'),res=res2, tt='segmentation'))
            s.layers.append(name=f'{name}_orange',layer=ngLayer((vesicles==2).astype('uint16'),res=res2, tt='segmentation'))
            s.layers.append(name=f'{name}_red',layer=ngLayer((vesicles==3).astype('uint16'),res=res2, tt='segmentation'))

            print(f"added all layers for {name}")
        else:
            print(f"file for {name} does not exist")

        del mask, vesicles
        cache.clear()
        gc.collect()


print(viewer)







