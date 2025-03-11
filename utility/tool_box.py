"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : This script defines all help functions.
"""

import os
import yaml
import torch
import argparse
from itertools import count


def adc_list_generate(data_path, output_file="adc_list.yaml"):
    """
    Generate the list of data from the given data path.

    Parameters:
        data_path (str): The path to the radar data.

    Returns:
        list: The list of data.
    """

    adc_list = []

    # Iterate through the folders in the data path
    for folder in os.listdir(data_path):
        entry = {
            "prefix": str(folder),
            "camera": "ZED",
            "config": "1843_azi & 1843_coherentEle",
            "radar": "AWR1843Boost"
        }
        adc_list.append(entry)
    
    # Save the list to a YAML file
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(adc_list, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print("ADC list generated successfully.")


def split_yaml(input_file, output_path, num_files):
    """
    Split a YAML file into multiple files based on the given file num.

    Parameters:
        input_file (str): The input YAML file to split.
        output_path (str): The path to save the output files.
        num_files (int): The number of files to split the data into.
    """

    with open(f"{input_file}.yaml", 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("YAML should contain a list of entries.")
    
    total_entries = len(data)
    batch_size = math.ceil(total_entries / num_files)

    # Split the data into batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        output_file = f"{output_path}/{input_file}_{i//batch_size + 1}.yaml"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(batch, f, allow_unicode=True, default_flow_style=False)


def reshape_fortran(x, shape):
    """
    Reshape a tensor in a Fortran-style (column-major order) while maintaining PyTorch's row-major default.

    Parameters:
        x (torch.Tensor): The input tensor to reshape.
        shape (tuple): The target shape in Fortran-style (column-major order).

    Returns:
        torch.Tensor: The reshaped tensor in the desired shape, maintaining Fortran-style ordering.
    """
    
    if len(x.shape) > 0:
        # Reverse the order of dimensions
        x = x.permute(*reversed(range(len(x.shape))))
    # Reshape and reverse the shape dimensions
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def peak_detect(input, gamma, sidelobeLevel_dB):
    """
    Detect peaks in the input data (input) based on the given gamma factor and sidelobe threshold in dB.

    Parameters:
    - input: 1D array or list of input signal data
    - gamma: Threshold factor for peak detection
    - sidelobeLevel_dB: The minimum required dB difference from the absolute maximum value to consider a peak valid.

    Returns:
    - peakVal: Tensor containing the values of the detected peaks
    - peakLoc: Tensor containing the locations (indices) of the detected peaks
    """

    device = torch.device(input.device)
    minVal, maxVal, maxLoc, maxLoc_r = torch.tensor(float('inf'), dtype=torch.float64, device=device), torch.tensor(0.0, device=device), 0, 0
    absMaxValue, locateMax, initStage, maxData, extendLoc = torch.tensor(0.0, device=device), False, True, torch.tensor([], device=device), 0

    N = input.shape[0]
    for i in count(0):
        if i >= N + extendLoc - 1:
            break
        i_loc = i % N  
        currentVal = input[i_loc]

        # Update maximum and minimum values
        absMaxValue = torch.max(absMaxValue, currentVal)
        if currentVal > maxVal:
            maxVal = currentVal
            maxLoc = i_loc
            maxLoc_r = i
        
        minVal = torch.min(minVal, currentVal)

        if locateMax:
            if currentVal < (maxVal / gamma):  # Peak found
                maxData = torch.cat((maxData, torch.tensor([[maxLoc, maxVal, i - maxLoc_r, maxLoc_r]], device=device)), dim=0)
                minVal = currentVal
                locateMax = False
        else:
            if currentVal > minVal * gamma:    # Valley found
                locateMax = True
                maxVal = currentVal
                if initStage:
                    extendLoc = i
                    initStage = False

    # Filter peaks based on sidelobe threshold
    absMaxValue_db = absMaxValue * (10 ** (-sidelobeLevel_dB / 10))
    maxData = maxData[maxData[:, 1] >= absMaxValue_db] if len(maxData) > 0 else maxData

    # Convert results to torch tensors
    peakVal, peakLoc = torch.tensor([], dtype=torch.float64, device=device), torch.tensor([], dtype=torch.long, device=device)
    if len(maxData) > 0:
        peakVal = maxData[:, 1].clone().detach().to(torch.float64)
        peakLoc = (maxData[:, 0] % N).clone().detach().to(torch.long)

    return peakVal, peakLoc


def parse_arguments():
    """
    Parse command-line arguments for the radar data conversion script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
        - data_root: Root path of dataset.
        - yaml_path: Path to the radar data file.
        - pipeline: Processing pipeline to use.
        - save: Whether to save the results to a file.
        - display: Whether to display the results.
    """

    parser = argparse.ArgumentParser(description='Convert radar data to PCD format.')
    parser.add_argument('--data_root', type=str, default="data/adc_data", help='Root path of dataset.')
    parser.add_argument('--yaml_path', type=str, default="adc_list", help='Path to the radar data file.')
    parser.add_argument('--pipeline', type=int, default=1, help='Processing pipeline to use.')
    parser.add_argument('--save', action='store_true', help='Whether to save the results to a file.')
    parser.add_argument('--display', action='store_true', help='Whether to display the results.')

    return parser.parse_args()