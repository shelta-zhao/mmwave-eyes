"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : Parses mmWave Studio config JSON files.
"""

import os
import json
import yaml
import numpy as np
from param_generate import generate_params


def get_num_frames(raw_data_path, data_size_one_frame):
    """
    Calculate the total number of frames and frames per binary file, and open the files.

    Parameters:
        raw_data_path (str): Binary file paths.
        data_size_one_frame (int): Size of one frame in bytes.

    Returns:
        tuple: (bin_file_frames, file_handles)
            - bin_file_frames: List of number of frames per binary file.
            - file_handles: List of file handles for opened binary files.
    """

    # Get all binary files in the directory
    bin_file_names = [f for f in os.listdir(raw_data_path) if f.endswith('.bin')]
    if len(bin_file_names) == 0:
        raise RuntimeError(f"No binary files found in the {raw_data_path}.")
    
    # Get the total number of frames and frames per binary file
    bin_file_frames, file_handles = [], []
    for _, bin_file_name in enumerate(bin_file_names):
        try:
            # Open the file
            bin_file_path = os.path.join(raw_data_path, bin_file_name)
            file_handle = open(bin_file_path, 'rb')
            file_handles.append(file_handle)

            # Get file size
            file_size = os.path.getsize(bin_file_path)

            # Calculate number of frames in the binary file
            num_frames = file_size // data_size_one_frame
            bin_file_frames.append(num_frames)

            # Check if the file has enough data for at least one frame
            if num_frames == 0:
                raise RuntimeError(f"Not enough data in binary file.")
            
        except Exception as e:
            raise RuntimeError(f"Error opening or processing file {bin_file_name}: {e}.")
    
    return bin_file_frames, file_handles


def load_one_frame(frame_idx, bin_file_frames, file_handles, data_size_one_frame):
    """
    Load one frame of data from the binary files.

    Parameters:
        frame_idx (int): Index of the frame to load, note that the index starts from 1.
        bin_file_frames (list): List of number of frames per binary file.
        file_handles (list): List of file handles for opened binary files.
        data_size_one_frame (int): Size of one frame in bytes.

    Returns:
        np.ndarray: Data for one frame.
    """

    # Check if the frame index is valid
    if frame_idx <= 0:
        raise ValueError("Frame index starts from 1.")

    # Determine which binary file contains the requested frame
    fid_idx, curr_frame_idx = -1, 0
    num_bin_files = len(bin_file_frames)
    for idx in range(num_bin_files):
        if frame_idx <= (bin_file_frames[idx] + curr_frame_idx):
            fid_idx = idx
            break
        curr_frame_idx += bin_file_frames[idx]

    if fid_idx == -1 or fid_idx >= num_bin_files:
        raise ValueError("Frame index out of range for the given binary files.")

    # Seek to the frame position in the identified binary file
    file_handle = file_handles[fid_idx]
    converted_frame_idx = frame_idx - curr_frame_idx - 1
    file_handle.seek(converted_frame_idx * data_size_one_frame, os.SEEK_SET)

    # Load one frame data
    try:
        # Read raw data and convert to float32
        raw_data = np.fromfile(file_handle, dtype=np.uint16, count=data_size_one_frame // 2).astype(np.float32)
        if len(raw_data) * 2 != data_size_one_frame:
            raise ValueError(f"Read incorrect data length: {len(raw_data)*2}, expected: {data_size_one_frame}")

        # Adjust values greater than 32768 to negative range
        time_domain_data = raw_data - (raw_data >= 2 ** 15) * 2 ** 16
        return time_domain_data
    except Exception as e:
        raise IOError(f"Error reading frame data: {e}")
    

if __name__  == "__main__":

    with open("data2parse.yaml", "r") as file:
        data = yaml.safe_load(file)
    raw_data_path = os.path.join("rawData/adcDatas", f"{data['prefix']}/{data['index']}")
    config_path = os.path.join("rawData/configs", data["config"])
    
    radar_params = generate_params(config_path, data['radar'])
    readObj = radar_params['readDataParams']

    bin_file_frames, file_handles = get_num_frames(raw_data_path, readObj['dataSizeOneFrame'])
    time_domain_data = load_one_frame(1, bin_file_frames, file_handles, readObj['dataSizeOneFrame'])
    print(type(time_domain_data))
    print(time_domain_data.shape)
    