# basic dependencies
import os
import sys
import glob
import time
# remote dependencies
import tqdm
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from connectomics.data.utils.data_io import readvol, savevol



##### ------------------------- #####
# Defined functions
#   - get_usage: boilerplate
#   - get_border: compute neuron boundary
#   - remove_dust: remove thin regions from neuron
#   - get_anchor: find seed for neuron boundary subregion
#   - get_region: expand from seed to get neuron boundary subregion
##### ------------------------- #####



##### ------------------------- #####
# get_usage: get the usage message
#
# outputs:
#   usage: the usage message
##### ------------------------- #####
def get_usage():
    usage = f"""
python {sys.argv[0]} neuron soma lv sv thres volume
    neuron: path to neuron mask
    soma: path to soma mask
    lv: path to large vesicle labels
    sv: path to small vesicle labels
    thres: threshold for how "deep" to penetrate to get small vesicles count.
        measured in voxels
    volume: desired volume for patch
    """
    return usage
##### ------------------------- #####



##### ------------------------- #####
# get_border: return the thres-voxel border of the neuron volume (excluding
#   the soma) 
#
# inputs:
#   neuron: the neuron volume, should be a binary three-dimensional np array
#       that
#       is isotropic or near-isotropic
#   soma: the soma volume, should be a binary three-dimensional np array that
#       is isotropic or near-isotropic
#   thres: the number of voxels that should be eroded/dilated, should be
#       an integer
#
# outputs:
#   neuron_border: the neuron border volume as a binary volume
##### ------------------------- #####
def get_border(neuron, soma, thres):
    # compute binary mask of the neuron interior, i.e. portions of neuron
    # that are more than thres voxels from the border
    neuron_interior = np.where(edt(neuron-soma)>=thres, 1, 0).astype(np.uint16)
    # compute valid neuron border by removing neuron interior and soma
    neuron_border = np.clip(neuron-neuron_interior, 0, 1)*(1-soma)
    return neuron_border
##### ------------------------- #####



##### ------------------------- #####
# remove_dust: perform binary erosion and dilation to remove neurites
#
# inputs:
#   vol: the volume to perform erosion/dilation on, should be a three-
#       dimensional np array that is isotropic or near-isotropic
#   thres: the number of voxels that should be eroded/dilated, should be
#       an integer
#
# outputs:
#   dusted: the dusted volume
##### ------------------------- #####
def remove_dust(vol, thres):
    # binary erosion
    eroded_vol = np.where(edt(vol)>=thres, 1, 0).astype(np.uint16)
    # binary dilation
    dusted = np.where(edt(1-eroded_vol)<=thres, 1, 0).astype(np.uint16)
    return dusted
##### ------------------------- #####



##### ------------------------- #####
# get_anchor: given a volume of valid anchor points, return valid coordinates
#   through sampling
#
# inputs:
#   vol: the volume of valid anchor points, should be a binary volume
#   num_samples: the number of samples to take, defaults to 1000000
#   progress: boolean value for whether tqdm progress should be shown, defaults
#       to True
#
# outputs:
#   z, y, x: valid coordiantes as integers
##### ------------------------- #####
def get_anchor(vol, num_samples=1000000, progress=True):
    for idx in tqdm.trange(num_samples) if progress else range(num_samples):
        z, y, x = np.floor(np.random.rand(3)*vol.shape).astype(np.uint16)
        if vol[z, y, x] == 1:
            return z, y, x
    raise Exception(f'Sampled {idx} times with no valid anchor')
##### ------------------------- #####



##### ------------------------- #####
# get_region: given a volume of valid anchor points, one specific anchor point,
#   and a desired volume, return a masked copy of the volume that contained a
#   subregion of the valid anchor points with least the target volume along
#   with some metadata
#
# inputs:
#   vol: the volume of valid anchor points, should be a binary volume
#   anchor: the supplied anchor point, should be an array of length three
#   target: the target volume for the subregion in voxels, should be an integer
#   max_radius: the maximum radius, defaults to 1000
#
# outputs:
#   region: the masked copy of the volume
#   region_volume: the volume, in voxels, of the region
#   radius: the final radius
##### ------------------------- #####
def get_region(vol, anchor, target, max_radius=1000):
    z, y, x = anchor
    initial_radius = np.floor(np.cbrt(target)).astype(np.uint16)
    for radius in range(initial_radius, initial_radius+max_radius):
        region_volume = neuron_border[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius].sum()
        if region_volume >= target:
            mask = np.zeros(vol.shape)
            mask[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius] = 1
            region = vol*mask
            return region, region_volume, radius
    raise Exception(f'Increased radius by {max_radius} without finding valid region')
##### ------------------------- #####


if __name__ == '__main__':
    # basic error management
    try:
        start_time = time.time()

        # make sure that right number of arguments passed
        assert len(sys.argv) == 7

        # organize metadata
        neuron, soma, lv, sv = sys.argv[1:5]
        thres = int(sys.argv[5])
        volume = int(sys.argv[6])

        # neuron, lv, and sv volumes are in resolution [30, 8, 8]. to approximate
        # isotropy, we dowsample the y and x dimensions by four times. soma volume
        # is already in [30, 32, 32]; no change is necessary
        neuron = np.clip(readvol(neuron), 0, 1)[:, ::4, ::4]
        soma = np.clip(readvol(soma), 0, 1)
        lv = readvol(lv)[:, ::4, ::4]
        sv = readvol(sv)[:, ::4, ::4]
    except Exception as e:
        print(get_usage())
        raise e

    # find valid region for non-soma sample
    print("Getting neuron border...")
    neuron_border = get_border(neuron, soma, thres)

    # remove (most of) soma and dendrites
    print("Removing dendrites and soma...")
    cropped = remove_dust(neuron-soma, 8)
    neuron_border = cropped * neuron_border
    savevol('neuron_border.h5', neuron_border)

    # randomly sample nonsoma_eroded to find a keypoint that works
    print("Getting anchor...")
    z, y, x = get_anchor(neuron_border)

    # expand region until sufficient volume
    print("Finding region...")
    region, region_volume, radius = get_region(neuron_border, [z, y, x], volume)

    # get number of small/large vesicles that overlap with the computed region;
    sv_count = len([x for x in np.unique(sv[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius]) if x != 0])
    lv_count = len([x for x in np.unique(lv[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius]) if x != 0])

    # get vesicle ID's and neuron border mask
    print(f'Border region volume is {region_volume}')
    print(f'Number of large/small vesicles is {lv_count}/{sv_count}, respectively.')
    if os.path.exists(f'thres_{thres:02}') == False:
        os.mkdir(f'thres_{thres:02}')
    savevol(f'thres_{thres:02}/border.h5', neuron_border.astype(np.uint16))
    savevol(f'thres_{thres:02}/lv.h5', lv.astype(np.uint16))
    savevol(f'thres_{thres:02}/sv.h5', sv.astype(np.uint16))
    print(f'Border mask and overlap with large/small vesicles stored in directory thres_{thres:02}/')
    
    # get time
    end_time = time.time()
    print(f'Process finished in {(end_time-start_time)/60:.1f} minutes.')
