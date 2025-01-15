"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : Load & Get regular raw radar data.
"""

import os
import yaml
import scipy
import numpy as np
import pandas as pd
import concurrent.futures
from datetime import datetime
from param_generate import generate_params


def get_timestamps(raw_data_path):
    """
    Extract the timestamp from the log file in the specified directory.

    Parameters:
        raw_data_path (str): Path to the directory containing log files.

    Returns:
        int: The extracted timestamp in milliseconds.
    """

    # Get log file names in the directory
    log_file_name = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')][0]
    
    # Read timestamps from the log file
    log_file = pd.read_csv(os.path.join(raw_data_path, log_file_name), skiprows=1, usecols=[0], on_bad_lines='skip', index_col=None)
    for _, row in log_file.iterrows():
        if row.str.contains('Capture start time').any():
            raw_timestamp = row.str.split(' - ').iloc[0][1]
            timestamp = datetime.strptime(raw_timestamp, "%a %b %d %H:%M:%S %Y")
            break

    # Return the timestamps in milliseconds
    return int(timestamp.timestamp() * 1000)


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
        np.ndarray: Data for one frame. Shape: (1, raw_data_per_frame).
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
        time_domain_data = raw_data - (raw_data >= 2**15) * 2**16
        return time_domain_data.reshape((1, -1)).astype(np.float32)
    except Exception as e:
        raise IOError(f"Error reading frame data: {e}")
    

def load_all_frames(bin_file_frames, file_handles, data_size_one_frame):
    """
    Load all frames of data from the binary files.

    Parameters:
        bin_file_frames (list): List of number of frames per binary file.
        file_handles (list): List of file handles for opened binary files.
        data_size_one_frame (int): Size of one frame in bytes.

    Returns:
        np.ndarray: Array containing all frame data. Shape: (num_frames, raw_data_per_frame).
    """

    all_frames_data = []  # To store data for all frames

    for fid_idx, num_frames_in_file in enumerate(bin_file_frames):
        file_handle = file_handles[fid_idx]
        try:
            # Seek to the beginning of the file
            file_handle.seek(0, os.SEEK_SET)

            # Load all frames from the current binary file
            raw_data = np.fromfile(file_handle, dtype=np.uint16, count=num_frames_in_file * (data_size_one_frame // 2)).astype(np.float32)

            if len(raw_data) * 2 != num_frames_in_file * data_size_one_frame:
                raise ValueError(f"Incorrect data length in file {fid_idx}: {len(raw_data) * 2}, expected: {num_frames_in_file * data_size_one_frame}")

            # Adjust values greater than 32768 to the negative range
            time_domain_data = raw_data - (raw_data >= 2**15) * 2**16

            # Reshape into frames and append to the list
            frames_data = time_domain_data.reshape((num_frames_in_file, -1)).astype(np.float32)
            all_frames_data.append(frames_data)

        except Exception as e:
            raise IOError(f"Error reading frames from file {fid_idx}: {e}")

    # Combine data from all binary files into a single array
    return np.vstack(all_frames_data)


def get_regular_data(readObj, time_domain_datas):
    """
    Get regular data from the time domain data.

    Parameters:
        readObj (dict): Dictionary containing radar parameters.
        time_domain_datas (np.ndarray): Time domain data. Shape: (num_frames, raw_data_per_frame).

    Returns:
        np.ndarray: Regular data. Shape: (num_frames, num_samples, num_chirps, num_rx, num_tx)
    """

    # Get the number of frames, chirps, and samples
    num_frames = time_domain_datas.shape[0]
    num_lane, ch_interleave, iq_swap = readObj['numLane'], readObj['chInterleave'], readObj['iqSwap']
    num_samples, num_chirps, num_tx, num_rx= readObj['numAdcSamplePerChirp'], readObj['numChirpsPerFrame'], readObj['numTxForMIMO'], readObj['numRxForMIMO']   

    def process_frame(frame_idx):
        # Get the raw data for one frame
        raw_data_complex = time_domain_datas[frame_idx, :].squeeze()

        # Reshape the time domain data based on the number of lanes
        raw_data_reshaped = np.reshape(raw_data_complex, (num_lane * 2, -1), order='F')
        raw_data_I = raw_data_reshaped[:num_lane, :].reshape(-1, order='F')
        raw_data_Q = raw_data_reshaped[num_lane:, :].reshape(-1, order='F')
        frame_data = np.column_stack((raw_data_I, raw_data_Q))

        # Swap I/Q if necessary
        frame_data[:, [0, 1]] = frame_data[:, [1, 0]] if iq_swap else frame_data[:, [0, 1]]

        # Combine I/Q data into complex data
        frame_data_complex = frame_data[:, 0] + 1j * frame_data[:, 1]

        # Reshape the complex data into regular data : (num_chirp, num_tx, num_rx, num_samples)
        frame_data_regular = frame_data_complex.reshape((num_samples * num_rx, num_chirps), order='F').T
        reshape_order = (num_chirps, num_rx, num_samples) if ch_interleave == 0 else (num_chirps, num_samples, num_rx)
        results = frame_data_regular.reshape(reshape_order, order='F').transpose(0, 2, 1)

        # Return the regular data
        return results.reshape((num_tx, -1, num_rx, num_samples), order='F').transpose(3, 1, 2, 0)

    # Process all frames in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
            regular_data = list(executor.map(process_frame, range(num_frames)))
    
    return np.stack(regular_data)


if __name__  == "__main__":

    with open("data2parse.yaml", "r") as file:
        data = yaml.safe_load(file)
    raw_data_path = os.path.join("rawData/adcDatas", f"{data['prefix']}/{data['index']}")
    config_path = os.path.join("rawData/configs", data["config"])
    
    radar_params = generate_params(config_path, data['radar'])
    readObj = radar_params['readObj']

    # Test timestamp extraction
    timestamp = get_timestamps(raw_data_path)
    print(f"Timestamp: {timestamp}")
    
    # Test frame loading
    bin_file_frames, file_handles = get_num_frames(raw_data_path, readObj['dataSizeOneFrame'])
    time_domain_data = load_one_frame(1, bin_file_frames, file_handles, readObj['dataSizeOneFrame'])
    # time_domain_datas = load_all_frames(bin_file_frames, file_handles, readObj['dataSizeOneFrame'])
    print(type(time_domain_data))
    print(time_domain_data.shape)

    # Test regular data
    regular_data = get_regular_data(readObj, time_domain_data)
    print(type(regular_data))
    print(regular_data.shape)

    # Save regular data as mat file
    scipy.io.savemat("regular_data_python.mat", {"regular_data": regular_data})
    