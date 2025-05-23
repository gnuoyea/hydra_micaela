# ves_analysis

## analysis

### metadata_and_kdtree
- `metadata_and_kdtree.py`: Convert format of dataset from binary mask to point cloud for each neuron, storing as a list of local coordinates of vesicle COMs and calculate statistics to store as corresponding attributes. Finally, construct kdtree from point cloud for density map, and export point cloud + metadata information into txt format for easy readability to dictionary format.
- `density_new.py`: Separate script to generate kdtree for density values if metadata is already generated.
- `merge_metadata.py`: Merging any added vesicle predictions from updated versions for adaptability and reduced computation.

### neuron_stitching
- `stitching_new.py`: Stitching together all pieces of adjacent neurons into the bounding box of a target neuron using global coordinate offsets; stitching two at a time to reduce memory consumption.

### vesicle_counts
- `pointcloud_soma_counts.py`: Easily generalizable method to return the number of vesicles within a neuronal region given as a binary mask (in this case, the somas of the neurons). Uses extracted values for each vesicle COM coordinate from the mask of the neuron region to determine which vesicles are within the region.
- `pointcloud_near_counts.py`: Extracts “near neuron” regions of interest via Euclidean Distance Transform for adjacent neuron pieces from stitched files (see `neuron_stitching/stitching_new.py`). Uses different thresholds for each vesicle type (see `vesicle_stats/lv_diameters.py`), then counts the number of vesicles of each type within their corresponding regions of interest, using the aforementioned method from `pointcloud_soma_counts.py`).
- `LV_type_counts.py, SV_type_counts.py`: Given exported lists (from `pointcloud_near_counts.py`) of segmentation IDs of vesicles within regions of interest and segID to type mappings, returns vesicle counts separated by type. Allows for adaptability and efficiency in cases of changes to type classifications.
- `slow_counts.py`: Slow, inefficient method—ignore if using point cloud metadata pipeline for vesicles. Manually counts the number of overlapping vesicles in a given region using vectorized binary mask operations (if both neurons and vesicles in form of binary masks).

### neuron_stats
- `surface_area.py`: Computes the surface area for a given binary mask of a neuron by generating a 1 voxel border using Euclidean distance transform and calculating the volume of the border region.

### vesicle_stats
- `updated_extract_stats.py`: Extract statistics from point cloud metadata txt format (`metadata_and_kdtree/metadata_and_kdtree.py`), save as a pandas dataframe, and export into spreadsheet format. Also export lists of volumes and diameters as txt files.
- `lv_diameters.py`: Finding diameter-based thresholds for near neuron counts, to be used in `vesicle_counts/pointcloud_near_counts.py`.
- `vesicle_volume_stats.py`: For calculating and exporting vesicle volumes only (not useful if using point cloud metadata format).
- `LUX2_density.py`: Calculating more specific stats for a particular region of interest within the LUX2 neuron.

## visualization
Currently UNUSED but functional alternate visualization methods, see `vesicleEM/ves_vis` for final visualization methods which are in use.

### neuroglancer_heatmap
- `heatmap.py`: Uses a Gaussian filter to calculate a heatmap image for vesicles within a given neuron, for visualization in Neuroglancer.
- `ng_heatmap_vis.py`: Neuroglancer rendering script to display a full dataset heatmap (from individual neuron heatmap images created using `heatmap.py`), assembled together by projecting into 2D and using coordinate offsets. Normalizing values and using a 4th color channel along with necessary rotations to align files for later color visualization.
- `ng_shader_script.glsl`: Shader script to plug into Neuroglancer visualization in order to render a gradient of colors based on values from 0.0 to 1.0.

### neuroglancer_types_map
- `types_visualization.py`: Generates “color coded” vesicle mask files given segmentation ID to type mappings by relabeling each vesicle to indicate its type.
- `types_ng.py`: Neuroglancer rendering script to display the full dataset of vesicles color coded by type, using previously generated mask files and using the offset feature to align neurons accurately.

### threshold_density_map
- `color_new.py`: Generates a heatmap using a Gaussian filter, then classifies/colors vesicles according to their density value through three thresholds as a simpler method if a continuous heatmap is not necessary.
- `color_new_ng.py`: Neuroglancer rendering script for thresholded heatmaps generated in `color_new.py`.
