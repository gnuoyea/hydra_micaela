import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
import matplotlib as plt
import matplotlib.cm as cm
import argparse
import gc

D0 = '/data/projects/weilab/dataset/hydra/results/'
D5 = '/data/rothmr/hydra/heatmaps/' #output dir

cache = {}

#downsample bc of the full ng vis
def load_data(name):
    if(name=="SHL17"): #fixing too large bbox
        with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f:
            cache["lv"] = np.array(f["main"][:, 4000:, :2400][:, ::2, ::2]) #30-16-16
        with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f:
            cache["sv"] = np.array(f["main"][:, 4000:, :2400][:, ::2, ::2]) #30-16-16

    else:
        with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f:
            cache["lv"] = np.array(f["main"][:][:, ::2, ::2]) #30-16-16
        with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f:
            cache["sv"] = np.array(f["main"][:][:, ::2, ::2]) #30-16-16

#when needing to ds more
def load_data_DS(name):
    with h5py.File(f"{D0}vesicle_big_{name}_30-8-8.h5", 'r') as f:
        cache["lv"] = f["main"][:, ::16, ::16] #30-64-64
    with h5py.File(f"{D0}vesicle_small_{name}_30-8-8.h5", 'r') as f:
        cache["sv"] = f["main"][:, ::16, ::16] #30-64-64


#write this for heatmap
def load_chunks(name, chunk_num, num_chunks):
    print(f"begin loading vesicles chunk #{chunk_num+1}") #bc zero indexing
    chunk_length = 0 #initialize for scope

    lv_path = f'{D0}vesicle_big_{name}_30-8-8.h5'
    with h5py.File(lv_path, 'r') as f:
        shape = f["main"].shape
        dtype = f["main"].dtype
        #calculate chunk_length (last chunk might be this plus some remainder if this doesn't divide evenly)
        chunk_length = (shape[1])//num_chunks #integer division, dividing up y axis length
        if(chunk_num!=num_chunks-1):
            cache["lv"] = f["main"][:, chunk_num*chunk_length:(chunk_num+1)*chunk_length, :]
        else: #case of the last chunk
            cache["lv"] = f["main"][:, chunk_num*chunk_length:, :] #go to end of the file - last chunk includes any leftover stuff
        print(f"done loading LV chunk #{chunk_num+1} of {num_chunks}") #bc zero indexing
    path = f'{D0}vesicle_small_{name}_30-8-8.h5'
    with h5py.File(path, 'r') as f:
        shape = f["main"].shape
        dtype = f["main"].dtype
        #calculate chunk_length (last chunk might be this plus some remainder if this doesn't divide evenly)
        chunk_length = (shape[1])//num_chunks #integer division, dividing up y axis length
        if(chunk_num!=num_chunks-1):
            cache["sv"] = f["main"][:, chunk_num*chunk_length:(chunk_num+1)*chunk_length, :]
        else: #case of the last chunk
            cache["sv"] = f["main"][:, chunk_num*chunk_length:, :] #go to end of the file - last chunk includes any leftover stuff
        print(f"done loading SV chunk #{chunk_num+1} of {num_chunks}") #bc zero indexing

    return chunk_length

def calculate_chunks(name, num_chunks):
    current_chunk = 0
    path = f"{D0}vesicle_big_{name}_30-8-8.h5"
    with h5py.File(path, 'r') as f:
        #output = np.zeros(shape=f["main"].shape, dtype=f["main"].dtype) #initialize the full output thing
        shape = f["main"].shape
        dtype = f["main"].dtype

    #init output file
    with h5py.File(f"{D5}{name}_heatmap.h5", 'w') as f:
        f.create_dataset('main', data=np.zeros(shape=shape,dtype=dtype))

    while(current_chunk!=num_chunks): #runs [num_chunk] times
        chunk_length = load_chunks(name, current_chunk, num_chunks)

        cache["lv"][cache["lv"] > 0] = 1
        cache["sv"][cache["sv"] > 0] = 1
        print("binary conversion done")

        #combine lv and sv
        cache["lv"] |= cache["sv"] #combine and save into cache["lv to not take up more memory"]
        cache["lv"] = cache["lv"].astype(np.int32) #convert bool to 1s and 0s
        del cache["sv"] #clear sv
        gc.collect()
        print("lv and sv combined")

        sigma = 10.0
        smoothed_density_map = gaussian_filter(cache["lv"].astype(np.float32), sigma=sigma)
        del cache["lv"]
        gc.collect()
        max_value = np.max(smoothed_density_map[smoothed_density_map > 0])  #find max
        if max_value > 0:
            smoothed_density_map[smoothed_density_map > 0] = (smoothed_density_map[smoothed_density_map > 0] / max_value * 255).astype(np.uint8)
        print("gaussian and normalization done")

        smoothed_density_map = smoothed_density_map.astype(np.float16) #reduce memory usage - change from float64 to float16...
        colormap = cm.plasma
        colored_density_map = (colormap(smoothed_density_map)[:, :, :, :3] * 255).astype(np.uint8)  #drop alpha
        print("final density map shape:", colored_density_map.shape)

        if(current_chunk!=num_chunks-1): #not last chunk
            with h5py.File(f"{D5}{name}_heatmap.h5", "a") as f:
                f["main"][:, y_start:(current_chunk + 1) * chunk_length, :] = heatmap_chunk

        else: #last chunk
            with h5py.File(f"{D5}{name}_heatmap.h5", "a") as f:
                f["main"][:, y_start:, :] = heatmap_chunk

        print(f"CHUNK {current_chunk} DONE \n")

        current_chunk+=1

    #save the output file to an h5
    print("done generating FULL heatmap file")
    return output


def calculate_density_map_WITH_SMOOTHING(name, downsample=False):
    if downsample: #ds by 8
        print(f"calculate_density_map running for {name}")

        load_data_DS(name)
        print("data loading and downsampling done")

        #change to in-place ops
        cache["lv"][cache["lv"] > 0] = 1
        cache["sv"][cache["sv"] > 0] = 1

        print("binary conversion done")

        #combine lv and sv
        cache["lv"] |= cache["sv"] #combine and save into cache["lv to not take up more memory"]
        cache["lv"] = cache["lv"].astype(np.int32) #convert bool to 1s and 0s
        del cache["sv"] #clear sv
        gc.collect()
        print("lv and sv combined")

        sigma = 10.0  #testing 2.0 to 20.0 - last was 10.0
        smoothed_density_map = gaussian_filter(cache["lv"].astype(np.float32), sigma=sigma)
        del cache["lv"]
        gc.collect()
     
        max_value = np.max(smoothed_density_map[smoothed_density_map > 0])  #find max
        if max_value > 0:
            #take out zero values so that normalization is not skewed
            smoothed_density_map[smoothed_density_map > 0] = (smoothed_density_map[smoothed_density_map > 0] / max_value * 255).astype(np.uint8)

        print("gaussian and normalization done")
        return smoothed_density_map

    else: #ds by 2 only
        print(f"calculate_density_map running for {name}")

        load_data(name)
        print("data loading and downsampling done")

        #change to in-place ops
        cache["lv"][cache["lv"] > 0] = 1
        cache["sv"][cache["sv"] > 0] = 1

        print("binary conversion done")

        #combine lv and sv
        cache["lv"] |= cache["sv"] #combine and save into cache["lv to not take up more memory"]
        cache["lv"] = cache["lv"].astype(np.int32) #convert bool to 1s and 0s
        del cache["sv"] #clear sv
        gc.collect()
        print("lv and sv combined")

        sigma = 10.0  #testing 2.0 to 20.0 - last was 10.0
        smoothed_density_map = gaussian_filter(cache["lv"].astype(np.float32), sigma=sigma)
        del cache["lv"]
        gc.collect()
     
        max_value = np.max(smoothed_density_map[smoothed_density_map > 0])  #find max
        if max_value > 0:
            #take out zero values so that normalization is not skewed
            smoothed_density_map[smoothed_density_map > 0] = (smoothed_density_map[smoothed_density_map > 0] / max_value * 255).astype(np.uint8)

        print("gaussian and normalization done")
        return smoothed_density_map


  
#takes in the normalized non color density map
def calculate_density_map_COLOR(density_map): 
    density_map = density_map.astype(np.float16) #reduce memory usage - change from float64 to float16...
    colormap = cm.plasma
    '''
    colored_density_map = colormap(density_map)  #rgba array
    colored_density_map = (colored_density_map[:, :, :, :3] * 255).astype(np.uint8)  #drop alpha
    '''
    colored_density_map = (colormap(density_map)[:, :, :, :3] * 255).astype(np.uint8)  #drop alpha

    print("final density map shape:", colored_density_map.shape)

    return colored_density_map

if __name__ == "__main__":
    #to_generate = ["KR4", "KR5", "KR6", "SHL55", "PN3", "LUX2", "SHL20", "KR11", "KR10", "RGC2", "KM4", "SHL17",
    #            "NET12", "NET10", "NET11", "PN7", "SHL18", "SHL24", "SHL28", "RGC7"]
    to_generate = ["NET10", "NET11", "SHL18"]


    chunking = False
    num_chunks = 4
    downsample = True

    if chunking:
        for name in to_generate:
            print(f"-----{name}-----")
            print(f"use chunking with {num_chunks} chunks")
            calculate_chunks(name, num_chunks)
            print(f"successfully saved {name}_heatmap \n")


    elif downsample:
        for name in to_generate:
            smoothed_density_map = calculate_density_map_WITH_SMOOTHING(name, downsample=True)
            color_density_map = calculate_density_map_COLOR(smoothed_density_map)

            with h5py.File(f"{D5}{name}_heatmap_DS.h5", 'w') as f:
                f.create_dataset('main', data=color_density_map)
                print(f"successfully saved {name}_heatmap_DS.h5 \n")


    else:
        for name in to_generate:
            smoothed_density_map = calculate_density_map_WITH_SMOOTHING(name)
            color_density_map = calculate_density_map_COLOR(smoothed_density_map)

            with h5py.File(f"{D5}{name}_heatmap.h5", 'w') as f:
                f.create_dataset('main', data=color_density_map)
                print(f"successfully saved {name}_heatmap \n")





