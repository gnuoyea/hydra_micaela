#author https://github.com/akgohain

import os
import gc
import math
import h5py
import numpy as np
from tqdm import tqdm
import argparse
from scipy import spatial
from concurrent.futures import ProcessPoolExecutor, as_completed

chunk_size = 25 #25            

ds_factor = 1             

voxel_dims = (30, 8 * ds_factor, 8 * ds_factor) 

sample_dir = '/home/rothmr/hydra/sample/'


#inpute into args to compute for all neurons
names_20 = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", 
            "RGC2", "KM4", "NET12", "NET10", "NET11", "PN7", "SHL18", 
            "SHL24", "SHL28", "RGC7", "SHL17"]


def process_chunk_range(params):
    start, end, file_path, ds_factor, name = params
    chunk_stats = {}
    with h5py.File(file_path, 'r', swmr=True) as f:
        dset = f["main"]

        chunk = dset[start:end, ::ds_factor, ::ds_factor]
        unique_labels = np.unique(chunk)
        for label in unique_labels:
            if label == 0:
                continue 
            mask = (chunk == label)
            coords = np.argwhere(mask)  
            if coords.size == 0:
                continue
            coords[:, 0] += start
            sum_coords = coords.sum(axis=0)
            count = coords.shape[0]
            if label in chunk_stats:
                chunk_stats[label]['sum'] += sum_coords
                chunk_stats[label]['count'] += count
            else:
                chunk_stats[label] = {'sum': sum_coords, 'count': count}
        del chunk
        gc.collect()
    return chunk_stats

def process_chunks(file_path, ds_factor, chunk_size, name, num_workers=None):
    chunk_stats = {}
    with h5py.File(file_path, 'r', swmr=True) as f:
        full_shape = f["main"].shape

    chunk_ranges = []
    for start in range(0, full_shape[0], chunk_size):
        end = min(start + chunk_size, full_shape[0])
        chunk_ranges.append((start, end, file_path, ds_factor, name))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk_range, params) for params in chunk_ranges]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks", ncols=100):
            result = future.result()
            for label, data in result.items():
                if label in chunk_stats:
                    chunk_stats[label]['sum'] += data['sum']
                    chunk_stats[label]['count'] += data['count']
                else:
                    chunk_stats[label] = data
    return chunk_stats


def compute_metadata(chunk_stats, voxel_dims):
    """
    Computes the center-of-mass (COM), volume, and radius for each vesicle.
    The volume is computed as: voxel_count * (30 * 64 * 64)
    The radius (in nm) is computed using the micaela/shulin's method:
       sqrt((voxel_count * (64*64)) / pi)
    """
    metadata = {}
    for label, stats in chunk_stats.items():
        com = stats['sum'] / stats['count']
        volume_nm = stats['count'] * (voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
        radius_nm = math.sqrt((stats['count'] * (voxel_dims[1] * voxel_dims[2])) / math.pi)
        metadata[label] = {
            'com': com,
            'volume_nm': volume_nm,
            'radius_nm': radius_nm
        }
    return metadata

def compute_density(metadata, voxel_dims, kd_radius=500):
    """
    Computes local vesicle density using a KDTree.
    COM coordinates are converted from voxel indices to physical units (nm) and
    then the density is calculated as: frequency / (kd_radius^2)
    The density values are also normalized.
    """
    com_list = []
    labels = []
    for label, data in metadata.items():
        com = data['com']
        # Convert voxel indices to physical coordinates (nm)
        physical_com = np.array([
            com[0] * voxel_dims[0],
            com[1] * voxel_dims[1],
            com[2] * voxel_dims[2]
        ])
        com_list.append(physical_com)
        labels.append(label)

    print("total num of ves: ", len(com_list))
    com_array = np.array(com_list)

    
    # Build the KDTree and query neighbors within kd_radius (nm)
    tree = spatial.KDTree(com_array)
    neighbors = tree.query_ball_tree(tree, kd_radius)
    frequency = np.array([len(n) for n in neighbors])
    density = frequency / (kd_radius ** 2)
    
    # Normalize the density values
    min_density = np.min(density)
    max_density = np.max(density)
    if max_density > min_density:
        normalized_density = (density - min_density) / (max_density - min_density)
    else:
        normalized_density = density

    # Add density info to metadata
    for i, label in enumerate(labels):
        metadata[label]['density'] = density[i]
        metadata[label]['normalized_density'] = normalized_density[i]
    return metadata

def process_vesicle_data(name, vesicle_type="lv"):
    """
    Processes vesicle data (either 'lv' or 'sv') using sequential chunking.
    Computes COM, volume, radius, and density via KDTree.
    The metadata is written out to a text file.
    
    Note: Re-enumerates labels to ensure uniqueness across LV and SV datasets.
    """
    file_prefix = "vesicle_big_" if vesicle_type == "lv" else "vesicle_small_"
    file_path = f"/data/projects/weilab/dataset/hydra/results/{file_prefix}{name}_30-8-8.h5"

    #if sample
    if(name=="sample"):
        if(vesicle_type == "lv"): #should only be lv
            file_path = f"{sample_dir}sample_data/7-13_pred_filtered.h5"
    
    print(f"Starting chunked processing for {vesicle_type} data of {name}...")
    chunk_stats = process_chunks(file_path, ds_factor, chunk_size, name)
    metadata = compute_metadata(chunk_stats, voxel_dims)
    metadata = compute_density(metadata, voxel_dims, kd_radius=500)
    
    # Re-enumerate labels to ensure uniqueness by prefixing with vesicle type
    unique_metadata = {}
    for label, data in metadata.items():
        new_label = f"{vesicle_type}_{label}"
        unique_metadata[new_label] = data
    metadata = unique_metadata
    
    # Ensure output directory exists
    output_dir = f"metadata/{name}/"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"metadata/{name}/{name}_{vesicle_type}_com_mapping.txt"

    if(name=="sample"):
        output_file = f"{sample_dir}sample_outputs/sample_com_mapping.txt"

    with open(output_file, "w") as f:
        for label, data in metadata.items():
            f.write(f"{data['com']}: ('{vesicle_type}', {label}, {data['volume_nm']}, {data['radius_nm']}, {data['density']}, {data['normalized_density']})\n")
    print(f"Chunked processing complete for {vesicle_type} data of {name}!")
    return metadata

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--which_neurons", type=str, help="all or sample?") #enter as "all" or "sample"
    args = parser.parse_args()
    which_neurons = args.which_neurons

    if(which_neurons=="sample"):
        lv_metadata = process_vesicle_data("sample", vesicle_type="lv")
        #only need LV for the near counts example pipeline

    elif(which_neurons=="all"):
        for name in names_20:
            lv_metadata = process_vesicle_data(name, vesicle_type="lv")
            sv_metadata = process_vesicle_data(name, vesicle_type="sv")




