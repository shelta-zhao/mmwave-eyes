"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : mmEyes-PCD pipeline to generate Point Cloud Data (PCD) from raw radar data.
"""

import os
import re
import sys
import yaml
import glob
import shutil
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.radarEyes_load import RadarEyesLoader, LidarDataProcessor
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor
from module.bp_process import BPProcessor


class mmEyesPCD:

    def __init__(self, device='cpu'):

        self.device = device
        
        self.config_path_azi = os.path.join("data/radar_config", "1843_azi")
        self.config_path_ele = os.path.join("data/radar_config", "1843_coherentEle")

    def run(self, adc_list, device, save=False, display=False):
        """
        Perform mmEyes PCD Pipeline.

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
        radarEyesLoader = RadarEyesLoader()
        config_azi = get_radar_params(self.config_path_azi, "AWR1843Boost", load=True)
        config_ele = get_radar_params(self.config_path_ele, "AWR1843Boost", load=True)
        for adc_data in adc_list:
            
            # Print the current data info
            print(f"\nProcessing data: {adc_data['prefix']}")

            # Generate regular data & radar params
            data_path = os.path.join("data/adc_data", f"{adc_data['prefix']}/{adc_data['index']}")
            
            # Check if the lidar data is processed
            if not os.path.exists(os.path.join("data/adc_data", adc_data['prefix'],"Lidar/Lidar_pcd")):
                self.process_lidar_data(os.path.join("data/adc_data", adc_data['prefix']))
                print(f"Lidar data processed successfully: {adc_data['prefix']}")

            # Perform data synchronization
            synchronized_data = radarEyesLoader.data_sync(data_path)
            
            # Check if the data is synchronized successfully
            if len(synchronized_data['radar_azi']['paths']) == 0 or len(synchronized_data['lidar']['paths']) == 0:
                print(f"Data synchronization failed. Please check the data : {adc_data['prefix']}.")
                continue

            # Perform mmEyes PCD pipeline for each frame
            radar_ele_all = np.load(os.path.join("data/adc_data", adc_data['prefix'],"1843_ele", "ADC", "all_frames.npy"))
            for frame_idx in tqdm(range(len(synchronized_data['radar_azi']['paths'])), desc="Processing frames", ncols=90):

                # Load data of different sensors
                radar_azi = radarEyesLoader.load_data(synchronized_data['radar_azi']['paths'][frame_idx], sensor='radar_azi')
                lidar = radarEyesLoader.load_data(synchronized_data['lidar']['paths'][frame_idx], sensor="lidar")


                # Perform Distributed Filter


                # Perform Polar Back Projection
                

                # radar_ele = radarEyesLoader.load_data(synchronized_data['radar_ele']['paths'][frame_idx], sensor="radar_ele")

    def process_lidar_data(self, data_path):
        """
        Process the raw lidar data to pcd data.
        
        Parameters:
            data_path (str): The path to the raw lidar data.
        """

        # Create folders for raw lidar data and pcd data
        lidarDataProcessor = LidarDataProcessor()
        src_folder = os.path.join(data_path, "Lidar")
        raw_folder = os.path.join(data_path, "Lidar/ADC")
        pcd_folder = os.path.join(data_path, "Lidar/Lidar_pcd")
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(pcd_folder, exist_ok=True)

        # Process each raw lidar data
        for file_path in tqdm(glob.glob(os.path.join(src_folder, "*.bin")), desc="Processing lidar frames", ncols=90):
            # Process the raw lidar bin file
            lidarDataProcessor.save_lidar(file_path)
            
            # Move the raw bin file to the ADC folder
            shutil.move(file_path, raw_folder)