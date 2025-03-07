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
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor
from module.bp_process import BPProcessor
from utility.visualizer_box import PCD_display, detect_display


class mmEyesPCD:

    def __init__(self):
        
        self.ADC_EXTENSIONS = ['.mat', '.MAT', 'bin', 'BIN', "jpg", "JPG","png","PNG", "npy"]
        pass