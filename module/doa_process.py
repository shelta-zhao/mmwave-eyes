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
            doa_results (torch.Tensor): The DOA estimation results.
        """

        for detection_point in detection_results:

            current_bin_val = detection_point['bin_val']

            self.DOA_beamformingFFT(current_bin_val)

            

        return 1
    
    def DOA_beamformingFFT(self, bin_val):
        
        # Get the DOA radar parameters
        apertureLen_azi = torch.max(self.D[:, 0]).item() + 1  
        apertureLen_ele = torch.max(self.D[:, 1]).item() + 1
        sig_2D = torch.zeros(apertureLen_azi, apertureLen_ele, dtype=torch.complex64)
        sig_2D[self.D[:, 0], self.D[:, 1]] = bin_val

        # Perform 2D Angle FFT
        doa_fft_result = torch.fft.fftshift(torch.fft.fftshift(torch.fft.fft2(sig_2D, s=(self.doa_fft_size, self.doa_fft_size), dim=[0, 1]), dim=0), dim=1)
        
        # Generate the doa estimation results
        obj_cnt, angleObj_est = 0, []
        spec_azi = sig_2D[self.D[self.antenna_azimuthonly, 0], self.D[self.antenna_azimuthonly, 1]].squeeze()
        _, peakLoc_azi = DOA_BF_PeakDet_loc(torch.abs(spec_azi))
        
        if apertureLen_ele == 1:
            # Azimuth array only, no elevation antennas
            for ind in peakLoc_azi:
                azi_est = torch.asin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * 180 / torch.pi
                if self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1]:
                    angleObj_est.append([azi_est, 0, ind, 0])
                    obj_cnt += 1
        else:
            # Azimuth and elevation angle estimation
            for ind in peakLoc_azi:
                spec_ele = torch.abs(doa_fft_result[ind, :])
                peakLoc_elev = DOA_BF_PeakDet_loc(obj_sidelobeLevel_dB_elev, spec_ele)
                
                for peak in peakLoc_elev:
                    # Calculate angles
                    azi_est = torch.arcsin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * -1 * 180 / torch.pi
                    ele_est = torch.arcsin(self.wz_vec[peak] / (2 * torch.pi * self.d)) * 180 / torch.pi
                    
                    if (self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1] and self.angles_DOA_ele[0] <= ele_est <= self.angles_DOA_ele[1]):
                        angleObj_est.append([azi_est, ele_est, ind, peak])
                        obj_cnt += 1

        return angleObj_est

        
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
    regular_data, timestamp = get_regular_data(data_path, radar_params['readObj'], '1', timestamp=True)

    # Perform Range & Doppler FFT
    fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], device)
    fft_output = fft_processor.run(regular_data)
    
    # Perform CFAR-CASO detection
    cfar_processor = CFARProcessor(radar_params['detectObj'], device)
    detection_results = cfar_processor.run(fft_output[0,:256,:,:,:])

    # Test DOA Estimation
    doa_processor = DOAProcessor(radar_params['DOAObj'], device)
    doa_output = doa_processor.run(detection_results)