import h5py
import os,sys
import numpy as np
from scipy.ndimage import zoom, binary_dilation
import neuroglancer
import socket
from contextlib import closing
import argparse
import yaml
from PIL import Image
import matplotlib.pyplot as plt

#full visualization, collapsed into 2d

D0 = '/data/projects/weilab/dataset/hydra/results/'
D5 = '/data/rothmr/hydra/heatmaps/'
bbox = np.loadtxt('/data/projects/weilab/dataset/hydra/mask_mip1/bbox.txt').astype(int) #for offsets


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

ip = 'localhost' #or public ip for sharable display
port = find_free_port()
neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
viewer=neuroglancer.Viewer()

resM = [32, 32, 30] #neuron mask
res_test = [16, 16, 30] #ves

#zyx
def ng_layer(data,res,oo=[0,0,0],tt='segmentation'):
    #data = data.transpose(2, 1, 0) #transpose from zyx to xyz
    data = data.transpose(1,2,0) #just this
    #data = data.transpose(0,2,1) #TESTING
    dim = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=res)
    return neuroglancer.LocalVolume(data, volume_type=tt, dimensions=dim, voxel_offset=oo)

#takes in 2d input and adds dim size 1 for z axis - for mask border
def ng_layer_2d(data, res, oo=[0,0,0], tt='segmentation'):
    data = np.expand_dims(data, axis=0) #add z axis
    #data = data.transpose(0,2,1)
    data = data.transpose(2,0,1) #just this 201
    #data = data.transpose(1,0,2) #TESTING
    dim = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=res)
    return neuroglancer.LocalVolume(data, volume_type=tt, dimensions=dim, voxel_offset=oo)


#for density map - process 4d input instead of 3d - xyz plus extra color channel
def ng_layer_edited(data, res, oo=[0, 0, 0], tt="image"):
    if data.ndim == 4:
        #shape is (d,h,w,c)
        data = data.transpose(2, 1, 0, 3)  #now (h,w,d,c) - from before
        #data = data.transpose(0, 1, 2, 3) #or this

        data = data.mean(axis=-1)  #average color channels -> (d,h,w) xzy

        #clip and change to uint8
        data = np.clip(data, 0, 255).astype(np.uint8)

    dim = neuroglancer.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=res)
    return neuroglancer.LocalVolume(data, volume_type=tt, dimensions=dim, voxel_offset=oo)

#extract the neuron border from collapsed 2d image for visualization
def extract_border(mask_2d):
    eroded_mask = binary_dilation(mask_2d, iterations=5) #more iterations for line thickness
    border = mask_2d ^ eroded_mask #XOR
    return border.astype(np.uint8)

def neuron_name_to_id(name):
    neuron_dict = read_yml('/data/projects/weilab/dataset/hydra/mask_mip1/neuron_id.txt') #switch to local path

    if isinstance(name, str):
        name = [name]
    return [neuron_dict[x] for x in name]

def read_yml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data


def get_offset(name): #returns in 30-8-8 and xyz
    nid = neuron_name_to_id(name)[0]
    bb = bbox[bbox[:,0]==nid, 1:][0] #bbox is in zyx
    if (name=="SHL17"):
        output = [bb[4], bb[2]+4000, bb[0]] #fixing too large bbox, change to xyz
    else:
        output = [bb[4], bb[2], bb[0]] #min coords, but change to xyz

    #take out z axis offset
    #output[1] = 0

    return output


def screenshot(path='temp.png', save=True, show=True, size=[4096, 4096]):
    ss = viewer.screenshot(size=size).screenshot.image_pixels
    if save:
        Image.fromarray(ss).save(path)
    if show:
        plt.imshow(ss)
        plt.show()


def visualize():
    names = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", "RGC2", "KM4", "SHL17", "PN7"]

    #once all heatmaps already generated
    for name in names:
        v_fname = f"{D5}{name}_heatmap.h5" #30-16-16
        with h5py.File(v_fname, "r") as f:
            vesicles = f["main"][:]
            #print("original size of heatmap:", vesicles.shape)

        m_fname = f"{D0}neuron_{name}_30-32-32.h5" #30-32-32
        if name=="SHL17":
            with h5py.File(f"/data/projects/weilab/dataset/hydra/results/neuron_{name}_30-32-32.h5") as f:
                mask = f["main"][:, 1000:, :600] #30-32-32, crop mask file
        else:
            with h5py.File(m_fname, "r") as f:
                mask = f["main"][:] #30-32-32

        offset = get_offset(name) #IN ZYX
        print("offset in 30-8-8 and xyz: ", offset) 

        res1_offset = [offset[0]/4, 0, offset[1]/4] #mask
        res2_offset = [offset[0]/4, 0, offset[1]/2] #heatmap


        with viewer.txn() as s:
            summed = np.sum(vesicles, axis=2) #collapse z-axis at angle of vis - sum up all densities
            density_map_normalized = (summed - np.min(summed)) / (np.max(summed) - np.min(summed))
            density_map_normalized = (density_map_normalized * 255).astype(np.uint8)
            #print("shape of collapsed density map: ", density_map_normalized.shape)

            density_map_normalized = np.rot90(density_map_normalized, k=3, axes=(0,1))

            s.layers.append(name=f'{name}_summed_density_map', layer=ng_layer(density_map_normalized, res=[30, 16, 16], tt='image', oo=res2_offset))
            print(f"added density map layer for {name}")

            mask_summed = np.any(mask, axis=2) #collapse z-axis - logical OR
            #print("shape of collapsed mask: ", mask_summed.shape)
            border = extract_border(mask_summed)

            border = np.rot90(border, k=3, axes=(0,1)) #rotate

            s.layers.append(name=f'{name}_mask',layer=ng_layer_2d(border,res=[30, 32, 32], tt='segmentation', oo=res1_offset)) #seg
            print(f"added mask border layer for {name}")


            #working 3d mask & 3d heatmap visualization for single neuron - can scroll heatmap layers to get true layer by layer overlay
            '''
            print("map shape: ", vesicles.shape)
            print("mask shape: ", mask.shape)
            s.layers.append(name=f'{name}_mask',layer=ng_layer(mask,res=[32, 32, 30], tt='segmentation')) #seg

            vesicles_transpose = vesicles.transpose(1,2,3,0) #3,0,1,2 #2,1,0,3
            print("shape of vesicles after transpose:", vesicles_transpose.shape)
            s.layers.append(name='density_map',layer=ng_layer_edited(vesicles,res=res_test, tt='image'))
            '''

    names2 = ["SHL24"] #DS by 4
    for name in names2:
        v_fname = f"{D5}{name}_heatmap.h5"
        with h5py.File(v_fname, "r") as f:
            vesicles = f["main"][:]
            #print("original size of heatmap:", vesicles.shape)

        m_fname = f"{D0}neuron_{name}_30-32-32.h5"
        with h5py.File(m_fname, "r") as f:
            mask = f["main"][:] #30-32-32

        offset = get_offset(name) #IN ZYX
        print("offset in 30-8-8 and xyz: ", offset) 

        res1_offset = [offset[0]/4, 0, offset[1]/4] #mask
        res2_offset = [offset[0]/4, 0, offset[1]/4] #heatmap


        with viewer.txn() as s:
            summed = np.sum(vesicles, axis=2) #collapse z-axis at angle of vis - sum up all densities
            density_map_normalized = (summed - np.min(summed)) / (np.max(summed) - np.min(summed))
            density_map_normalized = (density_map_normalized * 255).astype(np.uint8)

            density_map_normalized = np.rot90(density_map_normalized, k=3, axes=(0,1))

            s.layers.append(name=f'{name}_summed_density_map', layer=ng_layer(density_map_normalized, res=[30, 32, 32], tt='image', oo=res2_offset))
            print(f"added density map layer for {name}")

            mask_summed = np.any(mask, axis=2) #collapse z-axis - logical OR
            #print("shape of collapsed mask: ", mask_summed.shape)
            border = extract_border(mask_summed)

            border = np.rot90(border, k=3, axes=(0,1)) #rotate

            s.layers.append(name=f'{name}_mask',layer=ng_layer_2d(border,res=[30, 32, 32], tt='segmentation', oo=res1_offset)) #seg
            print(f"added mask border layer for {name}")



    names3 = ["SHL28", "RGC7"] #DS by 8
    for name in names3:
        v_fname = f"{D5}{name}_heatmap.h5"
        with h5py.File(v_fname, "r") as f:
            vesicles = f["main"][:]
            #print("original size of heatmap:", vesicles.shape)

        m_fname = f"{D0}neuron_{name}_30-32-32.h5"
        with h5py.File(m_fname, "r") as f:
            mask = f["main"][:] #30-32-32

        offset = get_offset(name) #IN ZYX
        print("offset in 30-8-8 and xyz: ", offset) 

        res1_offset = [offset[0]/4, 0, offset[1]/4] #mask
        res2_offset = [offset[0]/4, 0, offset[1]/8] #heatmap


        with viewer.txn() as s:
            summed = np.sum(vesicles, axis=2) #collapse z-axis at angle of vis - sum up all densities
            density_map_normalized = (summed - np.min(summed)) / (np.max(summed) - np.min(summed))
            density_map_normalized = (density_map_normalized * 255).astype(np.uint8)

            density_map_normalized = np.rot90(density_map_normalized, k=3, axes=(0,1))

            s.layers.append(name=f'{name}_summed_density_map', layer=ng_layer(density_map_normalized, res=[30, 64, 64], tt='image', oo=res2_offset))
            print(f"added density map layer for {name}")
 
            mask_summed = np.any(mask, axis=2) #collapse z-axis - logical OR
            #print("shape of collapsed mask: ", mask_summed.shape)
            border = extract_border(mask_summed)

            border = np.rot90(border, k=3, axes=(0,1)) #rotate

            s.layers.append(name=f'{name}_mask',layer=ng_layer_2d(border,res=[30, 32, 32], tt='segmentation', oo=res1_offset)) #seg
            print(f"added mask border layer for {name}")

    names4 = ["SHL18", "NET10", "NET11"] #DS by 16
    for name in names4:
        v_fname = f"{D5}{name}_heatmap.h5"
        with h5py.File(v_fname, "r") as f:
            vesicles = f["main"][:]
            #print("original size of heatmap:", vesicles.shape)

        m_fname = f"{D0}neuron_{name}_30-32-32.h5"
        with h5py.File(m_fname, "r") as f:
            mask = f["main"][:] #30-32-32

        offset = get_offset(name) #IN ZYX
        print("offset in 30-8-8 and xyz: ", offset) 

        res1_offset = [offset[0]/4, 0, offset[1]/4] #mask
        res2_offset = [offset[0]/4, 0, offset[1]/16] #heatmap

        with viewer.txn() as s:
            summed = np.sum(vesicles, axis=2) #collapse z-axis at angle of vis - sum up all densities
            density_map_normalized = (summed - np.min(summed)) / (np.max(summed) - np.min(summed))
            density_map_normalized = (density_map_normalized * 255).astype(np.uint8)

            density_map_normalized = np.rot90(density_map_normalized, k=3, axes=(0,1))

            s.layers.append(name=f'{name}_summed_density_map', layer=ng_layer(density_map_normalized, res=[30, 128, 128], tt='image', oo=res2_offset))
            print(f"added density map layer for {name}")

            mask_summed = np.any(mask, axis=2) #collapse z-axis - logical OR
            #print("shape of collapsed mask: ", mask_summed.shape)
            border = extract_border(mask_summed)

            border = np.rot90(border, k=3, axes=(0,1)) #rotate

            s.layers.append(name=f'{name}_mask',layer=ng_layer_2d(border,res=[30, 32, 32], tt='segmentation', oo=res1_offset)) #seg
            print(f"added mask border layer for {name}")


visualize()

print(viewer)








