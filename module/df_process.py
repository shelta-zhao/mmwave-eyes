"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define Distribute Filtering Algorithm Processor.
"""

import os
import sys
import torch
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor


class DFProcessor:

    def __init__(self, device):

        self.device = device

    def run(self, radar_datas, radar_params):
        
        # Generate all module instances
        fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], self.device)
        cfar_processor = CFARProcessor(radar_params['detectObj'], self.device)
        doa_processor = DOAProcessor(radar_params['DOAObj'], self.device)

        # Perform Range & Doppler FFT
        fft_output = fft_processor.run(radar_datas)
    
        # Perform CFAR-CASO detection
        for frameIdx in range(fft_output.shape[0]):
            detection_results, aa = cfar_processor.run(fft_output[frameIdx,:radar_params['detectObj']['rangeFFTSize'] // 2,:,:,:], frameIdx)

            # Perform DOA Estimation
            doa_results = doa_processor.run(detection_results)
            
            # Perform Distribute Filtering

            
            # Merge the DOA results
            if frameIdx == 0:
                point_cloud_data = doa_results
            else:
                point_cloud_data = np.concatenate((point_cloud_data, doa_results), axis=0)

        # Return the Point Cloud Data
        return point_cloud_data
        