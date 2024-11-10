import numpy as np
import h5py
from skimage.measure import label as skimage_label
import matplotlib as plt
import matplotlib.cm as cm


def calculate_density_map_NO_COLOR(nid):
    vesicles_fname = f"neuron{nid:02}_ALL_vesicles.h5"
    with h5py.File(vesicles_fname, 'r') as f:
        vesicle_data = f['main'][:]

    labeled_vesicles, num_vesicles = skimage_label(vesicle_data, return_num=True)

    print("calculate_density_map running")
    density_map = np.zeros(labeled_vesicles.shape, dtype=np.float32)
    mask = labeled_vesicles>0
    density_map[mask] = 1

    print("shape of vesicles density map: ", density_map.shape) #(188, 4096, 4096)

    #normalize to the range [0, 255]
    density_map = (density_map / np.max(density_map) * 255).astype(np.uint8)

    print("normalization done")
    return density_map
    

def calculate_density_map_COLOR(density_map): #takes in the normalized non color density map
    #downsampling (try to avoid if possible)
    density_map_ds = density_map[::2, ::2, ::2]
    #density_map_ds = density_map

    print("Density map shape:", density_map.shape)
    print("Downsampled density map shape:", density_map_ds.shape)

    colormap = cm.plasma
    colored_density_map = colormap(density_map_ds)  #rgba array
    colored_density_map = (colored_density_map[:, :, :, :3] * 255).astype(np.uint8)  #drop alpha
    print("Final density map shape:", colored_density_map.shape)

    return colored_density_map

if __name__ == "__main__":
    #example usage for new pipeline
    nid = 38
    density_map = calculate_density_map_NO_COLOR(nid)
    color_density_map = calculate_density_map_COLOR(density_map)

    fname = f"/local_copy/all_colormaps/neuron{nid}_colormap.h5"
    #visualization - ngs_colored_density_map_7-13.h5 works
    with h5py.File(fname, 'w') as f:
        f.create_dataset('main', data=color_density_map)
        print(f"successfully saved {fname}")



    



