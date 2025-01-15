"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : Perform radar process funciotns(3D-FFT, CFAR, DOA, etc).
"""

import os
import yaml
import torch
import scipy
import numpy as np
import pandas as pd
from param_generate import get_radar_params
from data_loader import get_regular_data


def adc_to_pcd(regular_data, radar_params, device, save=False, load=False):
    """
    Head Function: Perform procession chain to get Point Cloud Data(pcd) from raw regular data.

    Parameters:
        ababab
    
    Returns:
        abbaababa

    """

    if load:
        print('Hello')
    else:
        # Perform range_fft
        range_fft_output = range_fft(regular_data, radar_params['rangeFFTObj'], device)


def range_fft(input, rangeFFTObj, device):
    """
    Range FFT
    :param regular_data: regular data
    :param rangeFFTObj: range FFT object
    :return: range FFT data
    """

    # Get basic fft params
    radar_type, fft_size = rangeFFTObj['radarPlatform'], rangeFFTObj['rangeFFTSize']
    dc_on, win_on, scale_on, scale_factor = rangeFFTObj['dcOffsetCompEnable'], rangeFFTObj['rangeWindowEnable'], rangeFFTObj['FFTOutScaleOn'], rangeFFTObj['scaleFactorRange']

    # Convert input to tensor & Create output
    input = torch.tensor(input, dtype=torch.complex64).to(device)
    win_coeff = torch.tensor(np.hanning(input.shape[1] + 2)[1:-1], dtype=torch.float32).to(device)

    # Apply DC offset compensation
    input = input - torch.mean(input, dim=1, keepdim=True) if dc_on else input
    # Apply range-domain windowing
    input = input * win_coeff.view(-1, 1, 1, 1) if win_on else input
    # Perform FFT for each TX/RX chain
    fft_output = torch.fft.fft(input, n=fft_size, dim=1)
    # Apply scale factor
    fft_output = fft_output * scale_factor if scale_on else fft_output
    # Phase compensation for IWR6843ISK-ODS
    if radar_type == 'IWR6843ISK-ODS':
        fft_output[:, :, :, 1:3, :] *= torch.exp(-1j * torch.pi)

    # Return range fft result
    return fft_output.cpu() if device == 'cuda' else fft_output 

def doppler_fft(rangeFFT_output, dopplerFFTObj):
    """
    Doppler FFT
    :param rangeFFT_output: range FFT output
    :param dopplerFFTObj: doppler FFT object
    :return: doppler FFT data
    """
    doppler_fft_data = dopplerFFTObj.doppler_fft(rangeFFT_output)
    return doppler_fft_data

def angle_fft(rangeFFT_output, DOAObj):
    """
    Angle FFT
    :param rangeFFT_output: range FFT output
    :param angleFFTObj: angle FFT object
    :return: angle FFT data
    """
    angle_fft_data = angleFFTObj.angle_fft(rangeFFT_output)
    return angle_fft_data

def CFAR(range_fft_data, CFARObj):
    """
    CFAR
    :param range_fft_data: range FFT data
    :param CFARObj: CFAR object
    :return: CFAR data
    """
    CFAR_data = CFARObj.CFAR(range_fft_data)
    return CFAR_data


if __name__ == "__main__":
    
    # Parse data config & Get radar params
    with open("data2parse.yaml", "r") as file:
        data = yaml.safe_load(file)
    data_path = os.path.join("datas/adcDatas", f"{data['prefix']}/{data['index']}")
    config_path = os.path.join("datas/configs", data["config"])
    
    # Get radar params
    radar_params = get_radar_params(config_path, data['radar'], load=True)

    # Get regular raw radar data
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], '1', timestamp=True)

    # Test Range FFT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    range_fft_output = range_fft(regular_data, radar_params['rangeFFTObj'], device)
