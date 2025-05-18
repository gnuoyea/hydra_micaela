# ves_analysis

### metadata_and_kdtree
- `metadata_and_kdtree.py/`: Convert format of dataset from binary mask to point cloud for each neuron, storing as a list of local coordinates of vesicle COMs and corresponding attributes. Construct kdtree from point cloud for density map and export point cloud + metadata information into txt format for easy readability to dictionary format.
- `density_new.py/`: Separate script to generate kdtree for density values if metadata is already generated.
- `merge_metadata.py/`: Merging any added vesicle predictions from updated versions for adaptability and reduced computation.

### neuron_stitching
- `stitching_new.py/`: Stitching together all pieces of adjacent neurons into the bounding box of a target neuron using global coordinate offsets; stitching two at a time to reduce memory consumption.

### vesicle_counts
- `pointcloud_soma_counts.py/`: Easily generalizable method to return the number of vesicles within a neuronal region given as a binary mask (in this case, the somas of the neurons). Uses extracted values for each vesicle COM coordinate from the mask of the neuron region to determine which vesicles are within the region.
- `pointcloud_near_counts.py/`: Extracts “near neuron” regions of interest via Euclidean Distance Transform for adjacent neuron pieces from stitched files (see `neuron_stitching/stitching_new.py/`). Uses different thresholds for each vesicle type (see `vesicle_stats/lv_diameters.py/`), then counts the number of vesicles of each type within their corresponding regions of interest, using the aforementioned method from `pointcloud_soma_counts.py/`).
- `LV_type_counts.py/`: 
- `SV_type_counts.py/`: 
- `slow_counts.py/`: Slow, inefficient method—ignore if using point cloud metadata pipeline for vesicles. Manually counts the number of overlapping vesicles in a given region using vectorized binary mask operations (if both neurons and vesicles in form of binary masks).

### neuron_stats

### vesicle_stats

### neuroglancer_heatmap

### neuroglancer_types_map

### threshold_density_map
