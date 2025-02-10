"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define Back Projection Algorithm Processor, including FOV Mask.
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
from utility.tool_box import reshape_fortran


class BPProcessor:
    def __init__(self):
        pass

    def run(self, input):
        """
        Perform Back Projection Algorithm on the input data, GPU is needed to accelerate.

        Parameters:
            input (np.ndarray): The regular raw radar data.

        Returns:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.
        """
        pass

    def back_projection(self, rangefft_output):
        """
        Perform Back Projection Algorithm on the input data, GPU is needed to accelerate.

        Parameters:
            rangefft_output (np.ndarray): The output data from Range FFT.

        Returns:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.
        """
        pass

    def fov_mask(self, bp_output):
        """
        Perform Field of View (FOV) Mask on the input data.

        Parameters:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.

        Returns:
            fov_output (np.ndarray): The output data after Field of View (FOV) Mask.
        """
        pass

    def peak_detect(self, bp_output):
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
        