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
from utility.visualizer_box import PCD_display, fft_display


def rangeFFT(input):
    """
    Performs range FFT on the input data using the specified options.

    :param input: Input data array (numpy array)
    :param opt: Options including range_fftsize
    :return: Array with applied range FFT (numpy array)
    """
    fftsize = 128
    rangeWindowCoeffVec = np.hanning(input.shape[0] + 2)[1:-1]
    numLines = input.shape[1]
    numAnt = input.shape[2]
    out = np.zeros((fftsize, numLines, numAnt), dtype=complex)

    for i_an in range(numAnt):
        inputMat = input[:, :, i_an]
        # pdb.set_trace()
        # visData.plotHeatmap(np.abs(inputMat))
        inputMat = np.subtract(inputMat, np.mean(inputMat, axis=0))
        inputMat = inputMat * rangeWindowCoeffVec[:, None]
        fftOutput = np.fft.fft(inputMat, fftsize, axis=0)
        # pdb.set_trace()
        out[:, :, i_an] = fftOutput
        
    return out


def DopplerFFT(input):
    """
    Performs Doppler FFT on the input data using the specified options.

    :param input: Input data array (numpy array)
    :param opt: Options including doppler_fftsize
    :return: Array with applied Doppler FFT (numpy array)
    """
    fftsize = 128
    dopplerWindowCoeffVec = np.hanning(input.shape[1] + 2)[1:-1]
    numLines = input.shape[0]
    numAnt = input.shape[2]
    out = np.zeros((numLines, fftsize, numAnt), dtype=complex)

    for i_an in range(numAnt):
        inputMat = np.squeeze(input[:, :, i_an])
        inputMat = inputMat * dopplerWindowCoeffVec[None, :]
        fftOutput = np.fft.fft(inputMat, fftsize, axis=1)
        fftOutput = np.fft.fftshift(fftOutput, axes=1)
        out[:, :, i_an] = fftOutput

    return out


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

        radar_params = get_radar_params(config_path, adc_data['radar'], load=True)
        regular_data = np.fromfile(os.path.join(data_path, 'frame_2.bin'), dtype = "complex128").reshape((1, 128, 128, 4, 3))
        # Generate all module instances
        fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
        cfar_processor = CFARProcessor(radar_params['detectObj'], device)
        doa_processor = DOAProcessor(radar_params['DOAObj'], device)

        # Perform Range & Doppler FFT
        fft_output = fft_processor.run(regular_data)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # # # 假设 fft_out 是 [range_fft_size, doppler_fft_size] 大小
        radar_adc_data = regular_data[0,:,:,:,:].squeeze()
        rangeFFTOut = np.zeros((128, 128, 4, 3),dtype=complex)
        DopplerFFTOut = np.zeros((128, 128, 4, 3),dtype=complex)
        for i_tx in range(3):
            rangeFFTOut[:,:,:,i_tx] = rangeFFT(radar_adc_data[:,:,:,i_tx])
            rangeFFTOut[0:int(128*0.05),:,:,:] = 0
            rangeFFTOut[-int(128*0.1):,:,:,:] = 0
            DopplerFFTOut[:,:,:,i_tx] = DopplerFFT(rangeFFTOut[:,:,:,i_tx])
        
        import torch
        fft_out = torch.tensor(DopplerFFTOut[:,:,0,0])       
        fft_out = fft_output[0, :, :, 0, 0]
        range_fft_size, doppler_fft_size = fft_out.shape
        print(fft_out.shape)

        value, indices = torch.topk(abs(fft_out).view(-1), 5)
        indices_2d = torch.stack((indices // 128, indices % 128), dim=1)
        print(indices_2d    )
        print(value)
        fft_display(fft_out)
        aaaa
        # Perform CFAR-CASO detection
        for frameIdx in range(1):#tqdm(range(fft_output.shape[0]), desc="Processing frames"):
            # detection_results = cfar_processor.run(fft_output[:,:,:,:], frameIdx)
            detection_results = cfar_processor.run(fft_output[frameIdx,:,:,:,:], frameIdx)
            print(detection_results.shape)
            aaaa
            # Perform DOA Estimation
            doa_results = doa_processor.run(detection_results)

            # Merge the DOA results
            if frameIdx == 0:
                point_cloud_data = doa_results
            else:
                point_cloud_data = np.concatenate((point_cloud_data, doa_results), axis=0)
        
        # Display the PCD data if required
        if display:
            PCD_display(point_cloud_data)