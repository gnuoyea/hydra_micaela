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
    if(name=="SHL17"): #fixing too large bbox
        with h5py.File(f"color_coded/{name}_color_coded.h5", "r") as f:
            cache["vesicles"] = np.array(f["main"][:, 4000:, :2400][:,::8,::8]) #30-8-8
        with h5py.File(f"/data/projects/weilab/dataset/hydra/results/neuron_{name}_30-32-32.h5") as f:
            cache["mask"] = np.array(f["main"][:, 1000:, :600][:,::2,::2]) #30-32-32
        #load SV
        with h5py.File(f"/data/projects/weilab/dataset/hydra/results/vesicle_small_{name}_30-8-8.h5") as f:
            cache["sv"] = np.array(f["main"][:, 4000:, :2400][:,::8,::8]) #30-8-8

    else:
        with h5py.File(f"color_coded/{name}_color_coded.h5", "r") as f:
            cache["vesicles"] = np.array(f["main"][:][:,::8,::8])
        with h5py.File(f"/data/projects/weilab/dataset/hydra/results/neuron_{name}_30-32-32.h5") as f:
            cache["mask"] = np.array(f["main"][:][:,::2,::2])
        #load SV
        with h5py.File(f"/data/projects/weilab/dataset/hydra/results/vesicle_small_{name}_30-8-8.h5") as f:
            cache["sv"] = np.array(f["main"][:][:,::8,::8]) #30-8-8

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
    if (name=="SHL17"):
        output = [bb[0], bb[2]+4000, bb[4]] #fixing too large bbox
    else:
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
    names = ['KR4', 'KR5', 'KR6', 'SHL55', 'PN3', 'LUX2', 'SHL20', 'KR11', 'KR10', 'RGC2', 'KM4', 'NET12', 'SHL17', 'NET10', 'NET11', 'PN7', 'SHL18', 'SHL24', 'SHL28', 'RGC7']

    neuron_dict = read_yml('/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt') #switch to local path

    for name in names:
        path = f"color_coded/{name}_color_coded.h5"
        if(os.path.exists(path)):
            load_data(name) #cache["vesicles"] is updated to the current neuron's vesicle data
            print(f"done loading data for {name}")

            #do ds when loading

            vesicles = cache["vesicles"]
            mask = cache["mask"]
            sv = cache["sv"]

            offset = get_offset(name) #this is in 30-8-8?
            res1_offset = [offset[0], offset[1]/8, offset[2]/8] #mask 
            res2_offset = [offset[0], offset[1]/2, offset[2]/2] #lv
            res3_offset = [offset[0], offset[1]/4, offset[2]/4] #sv

            #note - downsampled everything to 30-64-64 for full vis so use res1 for everything

            s.layers.append(name=f'{name}_mask',layer=ngLayer(mask,res=res1, tt='segmentation', oo=res1_offset))
            s.layers.append(name=f'{name}_CV',layer=ngLayer((vesicles==1).astype('uint16'),res=res1, tt='segmentation', oo=res1_offset))
            s.layers.append(name=f'{name}_DV',layer=ngLayer((vesicles==2).astype('uint16'),res=res1, tt='segmentation', oo=res1_offset))
            s.layers.append(name=f'{name}_DVH',layer=ngLayer((vesicles==3).astype('uint16'),res=res1, tt='segmentation', oo=res1_offset))

            #add SV layer
            s.layers.append(name=f'{name}_small',layer=ngLayer(sv,res=res1, tt='segmentation', oo=res1_offset))

            print(f"added all layers for {name}")
        else:
            print(f"file for {name} does not exist")

        del mask, vesicles
        cache.clear()
        gc.collect()


print(viewer)







