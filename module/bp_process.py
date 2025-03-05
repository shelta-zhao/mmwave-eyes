"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define Back Projection Algorithm Processor, including FOV Mask.
"""

import os
import sys
import time
import yaml
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor
from utility.tool_box import reshape_fortran


class BPProcessor:
    def __init__(self, device, BPObj=None):
        """
        Initialize the Back Projection Processor with configurations.
        
        Parameters:
            device (str): The device to perform the computation on ('cpu' or 'cuda').
            BPObj (dict): Configuration for Back Projection
        """

        self.BPObj = BPObj
        self.device = device

        # Set default parameters for radar hetamap
        self.default = {
            'x_min': -3, 'x_max': 3, 'x_bins': 2000,
            'y_min': 0.4, 'y_max': 4.4, 'y_bins': 1000,
            'z_min': 0, 'z_max': 0.001, 'z_bins': 1
        }

        pass

    def run(self, input, positions, angles, timestamps, fft_processor):
        """
        Perform Back Projection Algorithm on the input data, GPU is needed to accelerate.

        Parameters:
            input (np.ndarray): The regular raw radar data.

        Returns:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.
        """

        # Convert all the input data to Tensor
        input = torch.tensor(input, dtype=torch.complex64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        angles = torch.tensor(angles, dtype=torch.float, device=self.device)

        # Select the first chirp from all frames    
        raw_data = input[:, :, 0:1, :, :]

        # Convert to Tensor and Perform Range FFT
        rangefft_out = fft_processor.range_fft(raw_data)
        
        # Perform Back Projection Algorithm
        bp_output = self.back_projection(rangefft_out, positions, angles, timestamps)
        pass
    
    def back_projection(self, rangefft_out, positions, angles, timestamps):
        """
        Perform Back Projection Algorithm on the input data, GPU is needed to accelerate.

        Parameters:
            rangefft_out (torch.Tensor): The output data from Range FFT.
            positions (np.ndarray): The positions of the radar.
            angles (np.ndarray): The angles of the radar.
            timestamps (np.ndarray): The timestamps of the radar.
            
        Returns:
            bp_output (torch.Tensor): The output data from Back Projection Algorithm.
        """

        # Create initial radar heatmap
        heatmap = self.init_heatmap()
        pass

    def init_heatmap(self, params=None):
        """
        Initialize the heatmap grid for back projection.

        Parameters:
            params (dict): The parameters for the heatmap grid.

        Returns:
            grid (torch.Tensor): The heatmap grid.
        """
        
        # Set default parameters if not provided
        if params is None:
            params = self.default

        # Extract basic parameters
        x_min, x_max, x_bins = params['x_min'], params['x_max'], params['x_bins']
        y_min, y_max, y_bins = params['y_min'], params['y_max'], params['y_bins']
        z_min, z_max, z_bins = params['z_min'], params['z_max'], params['z_bins']
        
        # Create the heatmap grid
        x_range = torch.linspace(x_min, x_max, x_bins, device=self.device)
        y_range = torch.linspace(y_min, y_max, y_bins, device=self.device)
        z_range = torch.linspace(z_min, z_max, z_bins, device=self.device)
        xx, yy, zz = torch.meshgrid(x_range, y_range, z_range)

        # Stack the grid into a single tensor
        grid = torch.stack((xx, yy, zz), dim=-1)
        
        return grid

    def fov_mask(self):
        """
        Perform Field of View (FOV) Mask on the input data.

        Parameters:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.

        Returns:
            fov_output (np.ndarray): The output data after Field of View (FOV) Mask.
        """
        pass

    def peak_detect(self):
        """
        Perform Peak Detection on the input data.

        Parameters:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.

        Returns:
            peak_output (np.ndarray): The output data after Peak Detection.
        """
        pass

    def bp_display(self, bp_output):
        """
        Visualizes the Back Projection output.

        Parameters:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.
        """
        pass
        