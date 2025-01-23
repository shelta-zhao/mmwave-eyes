"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define DOA Processor, including Angle FFT.
"""

import os
import sys
import yaml
import torch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.adc_load import get_regular_data
from fft_process import FFTProcessor
from cfar_process import CFARProcessor
from utility.tool_box import peak_detect
from utility.visualizer_box import PCD_display


class DOAProcessor:
    def __init__(self, DOAObj, device):
        """
        Initialize the DOAProcessor with DOA configurations.

        Parameters:
            DOAObj (dict): Configuration for DOA Estimation.
            device (str): The device to perform the computation on ('cpu' or 'cuda').
        """

        self.DOAObj = DOAObj
        self.device = device

        # Get the DOA radar parameters
        self.d = DOAObj['antDis']
        self.D = torch.tensor(DOAObj['D'])
        self.gamma = DOAObj['gamma']
        self.sidelobeLevel_dB = DOAObj['sidelobeLevel_dB']
        self.doa_fft_size = DOAObj['DOAFFTSize']
        self.angles_DOA_azi = DOAObj['angles_DOA_azi']
        self.angles_DOA_ele = DOAObj['angles_DOA_ele']
        self.antenna_azimuthonly = DOAObj['antenna_azimuthonly']
        self.wx_vec = torch.linspace(-torch.pi, torch.pi, self.doa_fft_size + 1)[:-1]
        self.wz_vec = torch.linspace(-torch.pi, torch.pi, self.doa_fft_size + 1)[:-1]
    
    def run(self, detection_results):
        """
        Run the DOA Estimation on the given detection results.

        Parameters:
            detection_results (list): The detection results from CFAR Processor.

        Returns:
            doa_results (list): The DOA estimation results.
        """

        # Initialize the list to store the output objects
        doa_estimate_result = []  

        for detection_point in detection_results:

            current_bin_val = detection_point['bin_val']
            DOAObj_est, doa_fft_result = self.DOA_beamformingFFT(current_bin_val)

            if len(DOAObj_est) > 0:
                for angle_obj in range(len(DOAObj_est)):  # Iterate through the columns of DOA_angles
                    doa_point = {}
                    
                    doa_point['frameIdx'] = detection_point['frameIdx']
                    doa_point['rangeInd'] = detection_point['rangeInd']
                    doa_point['dopplerInd'] = detection_point['dopplerInd']
                    doa_point['range'] = detection_point['range']
                    doa_point['doppler'] = detection_point['doppler']
                    doa_point['signalPower'] = detection_point['signalPower']
                    doa_point['noise_var'] = detection_point['noise_var']
                    doa_point['bin_val'] = detection_point['bin_val']
                    doa_point['estSNR'] = detection_point['estSNR']
                    
                    # Set the DOA angle and spectrum for the current object
                    doa_point['angles'] = DOAObj_est[angle_obj]
                    doa_point['spectrum'] = doa_fft_result 
                    
                    # Append the current object to the out list
                    doa_estimate_result.append(doa_point)

        # Perform coordinate transformation
        if not doa_estimate_result:
            return np.array([])

        point_cloud_data = np.zeros((len(doa_estimate_result), 14))  # Initialize the output array

        for iobj, result in enumerate(doa_estimate_result):
            # Coordinate transformation (x, y, z)
            azimuth, elevation = np.deg2rad(result['angles'][0]), np.deg2rad(result['angles'][1])
            point_cloud_data[iobj, 0] = result['frameIdx']                                    # Frame index
            point_cloud_data[iobj, 1] = iobj + 1                                              # Object index (start from 1)
            point_cloud_data[iobj, 2] = result['range'] * np.sin(azimuth) * np.cos(elevation) # X
            point_cloud_data[iobj, 3] = result['range'] * np.cos(azimuth) * np.cos(elevation) # Y
            point_cloud_data[iobj, 4] = result['range'] * np.sin(elevation)                   # Z

            # Distance, velocity, azimuth, elevation
            point_cloud_data[iobj, 5] = result['range']
            point_cloud_data[iobj, 6] = result['doppler']
            point_cloud_data[iobj, 7] = result['angles'][0]                                   # Azimuth
            point_cloud_data[iobj, 8] = result['angles'][1]                                   # Elevation

            # Noise, SNR
            point_cloud_data[iobj, 9] = result['signalPower']
            point_cloud_data[iobj, 10] = result['noise_var']
            point_cloud_data[iobj, 11] = result['estSNR']

            # RDM index
            point_cloud_data[iobj, 12] = result['rangeInd']
            point_cloud_data[iobj, 13] = result['dopplerInd']

        PCD_display(point_cloud_data)
        aaaa
        return point_cloud_data
    
    def DOA_beamformingFFT(self, bin_val):
        
        # Get the DOA radar parameters
        apertureLen_azi = torch.max(self.D[:, 0]).item() + 1  
        apertureLen_ele = torch.max(self.D[:, 1]).item() + 1
        sig_2D = torch.zeros(apertureLen_azi, apertureLen_ele, dtype=torch.complex64)
        sig_2D[self.D[:, 0], self.D[:, 1]] = bin_val

        # Perform 2D Angle FFT
        doa_fft_result = torch.fft.fftshift(torch.fft.fftshift(torch.fft.fft2(sig_2D, s=(self.doa_fft_size, self.doa_fft_size), dim=[0, 1]), dim=0), dim=1)
        
        # Generate the doa estimation results
        obj_cnt, DOAObj_est = 0, []
        spec_azi = sig_2D[self.D[self.antenna_azimuthonly, 0], self.D[self.antenna_azimuthonly, 1]].squeeze()
        _, peakLoc_azi = peak_detect(torch.abs(spec_azi), self.gamma, self.sidelobeLevel_dB[0])

        if apertureLen_ele == 1:
            # Azimuth array only, no elevation antennas
            for ind in peakLoc_azi:
                azi_est = torch.asin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * 180 / torch.pi
                if self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1]:
                    DOAObj_est.append([azi_est, 0, ind, 0])
                    obj_cnt += 1
        else:
            # Azimuth and elevation angle estimation
            for ind in peakLoc_azi:
                spec_ele = torch.abs(doa_fft_result[ind - 1, :])
                _, peakLoc_ele = peak_detect(spec_ele, self.gamma, self.sidelobeLevel_dB[1])
                for peak in peakLoc_ele:
                    print(ind, peak)
                    # Calculate angles
                    print(self.wx_vec[ind], self.wz_vec[peak])
                    azi_est = torch.arcsin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * -1 * 180 / torch.pi
                    ele_est = torch.arcsin(self.wz_vec[peak] / (2 * torch.pi * self.d)) * 180 / torch.pi
                    if (self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1] and self.angles_DOA_ele[0] <= ele_est <= self.angles_DOA_ele[1]):
                        print(azi_est, ele_est)
                        DOAObj_est.append([azi_est, ele_est, ind, peak])
                        obj_cnt += 1
            
        return DOAObj_est, doa_fft_result

        
if __name__ == "__main__":

    # Parse data config & Get radar params
    with open("adc_list.yaml", "r") as file:
        data = yaml.safe_load(file)
    data_path = os.path.join("data/adc_data", f"{data['prefix']}/{data['index']}")
    config_path = os.path.join("data/radar_config", data["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get radar params
    radar_params = get_radar_params(config_path, data['radar'], load=True)

    # Get regular raw radar data
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], 'all', timestamp=True)

    # Perform Range & Doppler FFT
    fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
    fft_output = fft_processor.run(regular_data)
    
    # Perform CFAR-CASO detection
    cfar_processor = CFARProcessor(radar_params['detectObj'], device)
    detection_results = cfar_processor.run(fft_output[25,:256,:,:,:], 0)

    # Test DOA Estimation
    doa_processor = DOAProcessor(radar_params['DOAObj'], device)
    doa_output = doa_processor.run(detection_results)