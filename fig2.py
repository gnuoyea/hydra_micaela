# basic dependencies
import os
import sys
import glob
# remote dependencies
import edt
import tqdm
import numpy as np
from connectomics.data.utils.data_io import readvol, savevol
# local dependencies
import counts

"""
Objective: Consider neuron 1, neuron 2 with appropriate neuron mask, soma mask,
and small vesicle labelling. Assume that we have access to a computed region of
overlap between the enlarged neuron 1 and neuron 2. Given a specific thickness,
compute the number of small vesicles in the overlapping region of neuron 2 and
a randomly selected non-soma region of neuron 2
"""

if __name__ == '__main__':
    # usage message
    usage = f"""
python {sys.argv[0]} neuron soma sv thres
    neuron: path to neuron mask
    soma: path to soma mask
    sv: path to small vesicle labels
    thres: threshold for how "deep" to penetrate to get small vesicles count.
        measured in nm.
    volume: desired volume for patch
    """
    if len(sys.argv) < 5:
        print(usage)
        exit(1)

    # metadata
    neuron, soma, sv = sys.argv[1:4]
    thres = sys.argv[4]
    volume = sys.argv[5]

    res = [32, 30, 30] # assume resolution, in [z, y, x] with units of nm 

    neuron = readvol(neuron) != 0
    soma = readvol(soma) != 0
    sv = readvol(sv)
    thres = int(thres)
    volume = int(volume)

    # find valid region for non-soma sample
    nonsoma_eroded = np.clip(edt.edt(neuron, order='F', anisotropy=res, parallel=0), 0, thres+1) * (soma == False)
    
    # randomly sample nonsoma_eroded to find a keypoint that works
    num_samples = 10000
    for idx in range(num_samples+1):
        if idx == num_samples:
            print(f'Sampled {num_samples} without valid sample.')
            exit(1)
        z, y, x = np.random.randint(nonsoma_eroded.shape[0]), np.random.randint(nonsoma_eroded.shape[1]), np.random.randint(nonsoma_eroded.shape[2])
        if nonsoma_eroded[z, y, x] > thres and nonsoma_eroded[z, y, x] < thres + res[2] * 2:
            break

    # expand region until sufficient volume
    neuron_border = neuron ^ (soma == 0) ^ (nonsoma_eroded != 0)
    radius = int(np.ceil(np.cbrt(volume)))
    while True:
        if neuron_border[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius].sum() >= volume:
            break
        else:
            radius += 1

    # crop all volumes and get number of small vesicles
    _ = np.zeros(neuron_border.shape); _[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius] = neuron_border[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius]; neuron_border = _
    sv = neuron_border * sv
    unique_sv = np.unique(sv[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius][1:])

    # get vesicle ID's and neuron border mask
    print(f'Border volume is {neuron_border[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius].sum()}')
    print(f'Border\'s rough bounding box is [{z-radius}:{z+radius}, {y-radius}:{y+radius}, {x-radius}:{x+radius}]')
    print(f'Number of small vesicles is {len(unique_sv)}. Labels are {unique_sv}')
    savevol('border_mask.h5', neuron_border)
    savevol('border_sv.h5', sv)
    print(f'Border mask and overlap with small vesicles stored at border_mask.h5 and border_sv.h5')
