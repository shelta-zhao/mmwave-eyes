"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : DREAM-PCD pipeline to generate Point Cloud Data (PCD) from raw radar data.
"""

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor
from utility.visualizer_box import PCD_display


def dream_pcd_pipeline(adc_list, device, save=False, display=False):
    """
    Generate Point Cloud Data (PCD) from raw radar data.

    Parameters:
        adc_list (str): The list of ADC data to be processed.
        device (str): The device to perform the computation on ('cpu' or 'cuda').
        save (bool): Whether to save the results to a file.
        display (bool): Whether to display the results.

    Returns:
        point_cloud_data (np.ndarray): The generated Point Cloud Data (PCD) from the raw radar data.
    """

    # Parse data config & Get radar params
    with open(f"{adc_list}.yaml", "r") as file:
        adc_list = yaml.safe_load(file)
    
    # Process each data in the list
    for adc_data in adc_list:
        
        # Print the current data info
        print(f"Processing data: {adc_data['prefix']} | {adc_data['config']} | {adc_data['radar']}")

        # Generate regular data & radar params
        data_path = os.path.join("data/adc_data", f"{adc_data['prefix']}/{adc_data['index']}")
        config_path = os.path.join("data/radar_config", adc_data["config"])
        
        radar_params = get_radar_params(config_path, adc_data['radar'])
        regular_data, _ = get_regular_data(data_path, radar_params['readObj'], 'all', timestamp=True)

        # Update the radar parameters
        radar_params['detectObj']['refWinSize'] = [8,4]
        radar_params['detectObj']['guardWinSize'] = [8,0]
        radar_params['detectObj']['K0'] = [5,3]
        radar_params['detectObj']['discardCellLeft'] = 10
        radar_params['detectObj']['discardCellRight'] = 20

        # Generate all module instances
        fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
        cfar_processor = CFARProcessor(radar_params['detectObj'], device)
        doa_processor = DOAProcessor(radar_params['DOAObj'], device)

        # Perform Range & Doppler FFT
        fft_output = fft_processor.run(regular_data)
        print(fft_output.shape)
        # Perform CFAR Detection
        cfar_output = cfar_processor.run(fft_output)
        print(cfar_output.shape)
        # Perform DOA Estimation
        doa_output = doa_processor.run(cfar_output)
        print(doa_output.shape)
        # Display the results
        if display:
            PCD_display(doa_output, radar_params['DOAObj']['angles_DOA_azi'], radar_params['DOAObj']['angles_DOA_ele'])