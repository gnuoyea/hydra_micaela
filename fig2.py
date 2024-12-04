# basic dependencies
import os
import sys
import glob
# remote dependencies
import edt
import tqdm
from scipy.ndimage import distance_transform_edt as edt
import numpy as np
from connectomics.data.utils.data_io import readvol, savevol
# local dependencies
import counts

"""
Objective: Consider neuron 1, neuron 2 with appropriate neuron mask, soma mask,
and smrge/all vesicle labelling. Assume that we have access to a computed region of
overlap between the enlarged neuron 1 and neuron 2. Given a specific thickness,
compute the number of small vesicles in the overlapping region of neuron 2 and
a randomly selected non-soma region of neuron 2
"""

if __name__ == '__main__':
    # usage message
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
    if len(sys.argv) < 6:
        print(usage)
        exit(1)

    # metadata
    neuron, soma, lv, sv = sys.argv[1:5]
    thres = sys.argv[5]
    volume = sys.argv[6]

    res = [30, 32, 32] # assume resolution, in [z, y, x] with units of nm

    neuron = np.clip(readvol(neuron), 0, 1).astype(np.uint32)
    soma = np.clip(readvol(soma), 0, 1)
    lv = readvol(lv)
    sv = readvol(sv)
    thres = int(thres)
    volume = int(volume)

    # find valid region for non-soma sample
    dt_neuron = edt(neuron-soma)
    eroded_neuron = np.where(dt_neuron >= thres, 1, 0).astype(np.uint16)
    neuron_border = np.clip(neuron - eroded_neuron, 0, 1) * (1 - soma)

    # randomly sample nonsoma_eroded to find a keypoint that works
    num_samples = 1000000
    for idx in range(num_samples+1):
        if idx == num_samples:
            print(f'Sampled {num_samples} without valid sample.')
            exit(1)
        z, y, x = np.random.randint(neuron_border.shape[0]), np.random.randint(neuron_border.shape[1]), np.random.randint(neuron_border.shape[2])
        if neuron_border[z, y, x] != 0:
            break

    # expand region until sufficient volume
    radius = int(np.ceil(np.cbrt(volume)))
    while True:
        if neuron_border[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius].sum() >= volume:
            break
        else:
            radius += 1

    # crop all volumes and get number of small vesicles
    _ = np.zeros(neuron_border.shape); _[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius] = neuron_border[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius]; neuron_border = _
    lv = neuron_border * lv
    sv = neuron_border * sv
    unique_lv = np.unique(lv[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius][1:])
    unique_sv = np.unique(sv[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius][1:])

    # get vesicle ID's and neuron border mask
    print(f'Border volume is {neuron_border[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius].sum()}')
    print(f'Border\'s rough bounding box is [{z-radius}:{z+radius}, {y-radius}:{y+radius}, {x-radius}:{x+radius}]')
    print(f'Number of large vesicles is {len(unique_lv)-1}. Labels are {unique_lv[1:]}')
    print(f'Number of small vesicles is {len(unique_sv)-1}. Labels are {unique_sv[1:]}')
    savevol(f'border{thres:02}_mask.h5', neuron_border.astype(np.uint16))
    savevol(f'border{thres:02}_lv.h5', lv.astype(np.uint16))
    savevol(f'border{thres:02}_sv.h5', sv.astype(np.uint16))
    print(f'Border mask and overlap with [large/small] vesicles stored at border_mask.h5 and border_[lv/sv].h5')
