"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Traditon pipline to generate Point Cloud Data (PCD) from raw radar data.
"""

import os
import sys
import yaml
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor


def adc_to_pcd(radar_params_file, adc_data_file, doa_config_file, device):
    """
    Generate Point Cloud Data (PCD) from raw radar data.

    Parameters:
        radar_params_file (str): The path to the radar parameters configuration file.
        adc_data_file (str): The path to the raw ADC data file.
        doa_config_file (str): The path to the DOA configuration file.
        device (str): The device to perform the computation on ('cpu' or 'cuda').

    Returns:
        pcd_data (list): The generated Point Cloud Data (PCD) from the raw radar data.
    """

    # Load the radar parameters
    radar_params = get_radar_params(radar_params_file)

    # Load the raw ADC data
    adc_data = get_regular_data(adc_data_file, radar_params)

    # Load the DOA configuration
    with open(doa_config_file, 'r') as f:
        DOAObj = yaml.safe_load(f)

    # Initialize the FFT Processor
    fft_processor = FFTProcessor(radar_params, device)

    # Perform the FFT processing
    fft_results = fft_processor.run(adc_data)

    # Initialize the CFAR Processor
    cfar_processor = CFARProcessor(radar_params, device)

    # Perform the CFAR processing
    cfar_results = cfar_processor.run(fft_results)

    # Initialize the DOA Processor
    doa_processor = DOAProcessor(DOAObj, device)

    # Perform the DOA processing
    doa_results = doa_processor.run(cfar_results)

    return doa_results
