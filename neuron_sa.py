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
        neuron= sys.argv[1]
        # neuron is in resolution [30, 8, 8]. to approximate isotropy, we
        # dowsample the y and x dimensions by four times
        neuron = np.clip(readvol(neuron), 0, 1)[:, ::4, ::4]
    except Exception as e:
        raise e

    neuron_edt = edt(neuron)
    neuron_border = ((neuron_edt>0) * (neuron_edt<=1)).astype(np.uint16)
    voxels = int(neuron_border.sum())

    print(f'surface area (sq. nm): {voxels*32**2}')
