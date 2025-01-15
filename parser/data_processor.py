"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : Load & Get regular raw radar data.
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from param_generate import get_radar_params
from data_loader import get_regular_data


def range_fft(input, rangeFFTObj, device):
    """
    Range FFT
    :param regular_data: regular data
    :param rangeFFTObj: range FFT object
    :return: range FFT data
    """
    # 'dopplerWindowCoeff': np.hanning(num_chirps_per_vir_ant)[:(num_chirps_per_vir_ant + 1) // 2],
    # Get basic Param & Convert input to tensor
    num_chirps, num_rx, num_tx = input.shape
    radar_type, fft_size = rangeFFTObj['radarPlatform'], rangeFFTObj['rangeFFTSize']
    win_on, win_coeff, scale_on, scale_factor = rangeFFTObj['rangeWindowEnable'], rangeFFTObj['rangeWindowCoeff'], rangeFFTObj['FFTOutScaleOn'], rangeFFTObj['scaleFactorRange']
    win_coeff = win_coeff.to(device)

    # Convert input to tensor & Create output
    input = torch.tensor(input, dtype=torch.complex64).to(device)
    output = torch.zeros((fft_size, num_chirps, num_rx, num_tx), dtype=torch.complex64).to(device)

    for tx in range(num_tx):
        for rx in range(num_rx):

            data = input[:, :, rx, tx].squeeze()

            # DC offset compensation
            data -= torch.mean(data, dim=0, keepdim=True)

            # Apply range-domain windowing
            if win_on:
                data = data * win_coeff.view(-1, 1)

            # Perform FFT
            fft_output = torch.fft.fft(data, n=fft_size, dim=0)

            if scale_on:
                fft_output *= scale_factor

            # Phase compensation for IWR6843ISK-ODS
            if radar_type == 'IWR6843ISK-ODS' and rx in [1, 2]:
                fft_output = fft_output * torch.exp(-1j * torch.pi)

            output[:, :, rx, tx] = fft_output
        
    return output

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
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], 'all', timestamp=True, load=True)
    print(regular_data.shape)
    aaaa

    # Test Range FFT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Hello Dislab_mmwavePCD!")