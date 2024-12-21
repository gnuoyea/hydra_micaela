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

if __name__ == '__main__':
    try:
        neuron= sys.argv[1] # should have resolution [30, 32, 32]
        neuron = np.clip(readvol(neuron), 0, 1)# [:, ::4, ::4]
    except Exception as e:
        raise e

    neuron_edt = edt(neuron)
    neuron_border = ((neuron_edt>0) * (neuron_edt<=1)).astype(np.uint16)
    voxels = int(neuron_border.sum())

    print(f'{neuron} surface area (sq. nm): {voxels*32**2}')
