"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define DOA Processor, including Angle FFT.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utility.tool_box import peak_detect


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
        self.antenna_azimuthonly = DOAObj['antenna_azimuthonly']

    def run(self, detection_results, save=False, load=False, denoise=False):
        """
        Run the DOA Estimation on the given detection results.

        Parameters:
            detection_results (list): The detection results from CFAR Processor.
            save (bool): Whether to save the results to a file.
            load (bool): Whether to load the results from a file.
            denoise (bool): Whether to remove noise from the DOA estimation results.

        Returns:
            point cloud data (np.ndarray): The DOA estimation results. 
            1. x: X coordinate
            2. y: Y coordinate
            3. z: Z coordinate
            4. signalPower: Signal power
            5. frameIdx: Frame index
            6. objectIdx: Object index
            7. distance: Distance
            8. velocity: Velocity
            9. azimuth: Azimuth angle
            10. elevation: Elevation angle

            11. noise_var: Noise variance
            12. estSNR: Estimated SNR
            13. rangeInd: Range index
            14. dopplerInd: Doppler index
        """

        # Initialize the list to store the output objects
        doa_estimate_result = []  
        for detection_point in detection_results:
            
            # Perform DOA estimation
            num_elements = detection_point[8:].numel() // 2
            current_bin_val = torch.complex(detection_point[8:8 + num_elements], detection_point[8 + num_elements:])
            DOAObj_est, doa_fft_result = self.DOA_beamformingFFT(current_bin_val)

            if len(DOAObj_est) > 0:
                for angle_obj in range(len(DOAObj_est)):  # Iterate through the columns of DOA_angles
                    doa_point = {}
                    
                    doa_point['frameIdx'] = detection_point[0]
                    doa_point['rangeInd'] = detection_point[1]
                    doa_point['range'] = detection_point[2]
                    doa_point['dopplerInd'] = detection_point[3]
                    doa_point['doppler'] = detection_point[4]
                    doa_point['noise_var'] = detection_point[5]
                    doa_point['signalPower'] = detection_point[6]
                    doa_point['estSNR'] = detection_point[7]
                    doa_point['bin_val'] = current_bin_val
                    
                    # Set the DOA angle and spectrum for the current object
                    doa_point['angles'] = DOAObj_est[angle_obj]
                    doa_point['spectrum'] = doa_fft_result 
                    
                    # Append the current object to the out list
                    doa_estimate_result.append(doa_point)

        # Perform coordinate transformation
        if not doa_estimate_result:
            return np.array([])
        
        # Remove noise
        if denoise:
            doa_estimate_result = self.remove_noise(doa_estimate_result)

        # Initialize the output array
        point_cloud_data = np.zeros((len(doa_estimate_result), 14))

        for iobj, result in enumerate(doa_estimate_result):
            # Coordinate transformation (x, y, z)
            azimuth, elevation = np.deg2rad(result['angles'][0].item()), np.deg2rad(result['angles'][1].item())
            point_cloud_data[iobj, 0] = result['range'] * np.sin(azimuth) * np.cos(elevation) # X
            point_cloud_data[iobj, 1] = result['range'] * np.cos(azimuth) * np.cos(elevation) # Y
            point_cloud_data[iobj, 2] = result['range'] * np.sin(elevation)                   # Z
            point_cloud_data[iobj, 3] = result['signalPower']                                 # Signal power
            point_cloud_data[iobj, 4] = result['frameIdx']                                    # Frame index
            point_cloud_data[iobj, 5] = iobj + 1                                              # Object index (start from 1)

            # Distance, velocity, azimuth, elevation
            point_cloud_data[iobj, 6] = result['range']
            point_cloud_data[iobj, 7] = result['doppler']
            point_cloud_data[iobj, 8] = result['angles'][0]                                   # Azimuth
            point_cloud_data[iobj, 9] = result['angles'][1]                                   # Elevation

            # Noise, SNR
            point_cloud_data[iobj, 10] = result['noise_var']
            point_cloud_data[iobj, 11] = result['estSNR']

            # RDM index
            point_cloud_data[iobj, 12] = result['rangeInd']
            point_cloud_data[iobj, 13] = result['dopplerInd']

        return point_cloud_data
    
    def DOA_beamformingFFT(self, bin_val):
        """
        Perform DOA estimation using 2D Angle FFT.

        Parameters:
            bin_val (torch.Tensor): The input signal data.
        
        Returns:
            DOAObj_est (np.ndarray): The DOA estimation results.
            angle_sepc_2D_fft (np.ndarray): The 2D Angle FFT result.
        """
        
        # Get the DOA radar parameters
        apertureLen_azi = torch.max(self.D[:, 0]).item() + 1  
        apertureLen_ele = torch.max(self.D[:, 1]).item() + 1
        sig_2D = torch.zeros(apertureLen_azi, apertureLen_ele, dtype=torch.complex64, device=self.device)
        sig_2D[self.D[:, 0], self.D[:, 1]] = bin_val

        # Perform 2D Angle FFT
        angle_sepc_1D_fft = torch.fft.fftshift(torch.fft.fft(sig_2D, n=self.doa_fft_size, dim=0), dim = 0)
        angle_sepc_2D_fft = torch.fft.fftshift(torch.fft.fft(angle_sepc_1D_fft, n=self.doa_fft_size, dim=1), dim=1)

        # Generate the doa estimation results
        obj_cnt, DOAObj_est = 0, torch.tensor([], device=self.device)
        spec_azi = torch.abs(angle_sepc_1D_fft[:, self.antenna_azimuthonly])
        _, peakLoc_azi = peak_detect(torch.abs(spec_azi), self.gamma, self.sidelobeLevel_dB[0])

        if apertureLen_ele == 1:
            # Azimuth array only, no elevation antennas
            for ind in peakLoc_azi:
                azi_est = torch.asin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * 180 / torch.pi
                if self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1]:
                    DOAObj_est = torch.cat((DOAObj_est, torch.tensor([azi_est, 0, ind, 0], device=self.device).unsqueeze(0)))
                    obj_cnt += 1
        else:
            # Azimuth and elevation angle estimation
            for ind in peakLoc_azi:
                spec_ele = torch.abs(angle_sepc_2D_fft[ind, :])
                _, peakLoc_ele = peak_detect(spec_ele, self.gamma, self.sidelobeLevel_dB[1])

                for peak in peakLoc_ele:
                    # Calculate angles
                    azi_est = torch.arcsin(self.wx_vec[ind] / (2 * torch.pi * self.d)) * 180 / torch.pi
                    ele_est = torch.arcsin(self.wz_vec[peak] / (2 * torch.pi * self.d)) * 180 / torch.pi

                    if (self.angles_DOA_azi[0] <= azi_est <= self.angles_DOA_azi[1] and self.angles_DOA_ele[0] <= ele_est <= self.angles_DOA_ele[1]):
                        DOAObj_est = torch.cat((DOAObj_est, torch.tensor([azi_est, ele_est, ind, peak], device=self.device).unsqueeze(0)))
                        obj_cnt += 1
            
        return DOAObj_est.cpu().numpy(), angle_sepc_2D_fft.cpu().numpy()

    def remove_noise(self, doa_estimate_result, angle_threshold=4):
        """
        Remove noise from the DOA estimation results.

        Parameters:
            doa_estimate_result (list): The DOA estimation results.
            angle_threshold (int): The threshold for angle difference.
        
        Returns:
            doa_result_filtered (list): The filtered DOA estimation results.
        """

        # Remove noise from the DOA estimation results
        doa_result_filtered = []
        for i, point1 in enumerate(doa_estimate_result):
            range1, snr1 = point1['rangeInd'], point1['estSNR']

            for j, point2 in enumerate(doa_estimate_result):
                if i == j:
                    continue

                range2, snr2 = point2['rangeInd'], point2['estSNR']
                az_diff = abs(point1['angles'][2] - point2['angles'][2])
                el_diff = abs(point1['angles'][3] - point2['angles'][3])

                # Compare the two points
                if az_diff <= angle_threshold and el_diff <= angle_threshold and range1 > range2 and snr1 < snr2:
                    break
            else:
                doa_result_filtered.append(point1)

        return doa_result_filtered
    
    def PCD_display(self, point_cloud_data):
        """
        Plot the point cloud data in a 3D scatter plot.

        Parameters:
        - point_cloud_data: A Numpy.ndarray of shape (N, 14), where:
            - Columns 0, 1, 2 are X, Y, Z coordinates.
        """

        # Extract X, Y, Z coordinates
        x = point_cloud_data[:, 0]
        y = point_cloud_data[:, 1]
        z = point_cloud_data[:, 2]
        velocity = point_cloud_data[:, 6]

        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color based on frame indices
        scatter = ax.scatter(x, y, z, c=velocity, cmap='viridis', s=20, alpha=0.8)

        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Velocity', rotation=270, labelpad=15)
        scatter.set_clim(-3, 3)

        # Set axis labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')

        # Set title and show
        ax.set_title('3D Point Cloud')
        plt.show()
