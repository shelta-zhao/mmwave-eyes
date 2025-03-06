"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define Back Projection Algorithm Processor, including FOV Mask.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

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

        # self.BPObj = BPObj
        self.BPObj = {
            "antDis" : 0.002,
            "lambda": 0.003843689942344651,
            "rangebinSize":0.09375,
            "frame_rate": 200,
            "TX_position":[[0, 0, 0],[-1, 0, 2],[0, 0, 4]], 
            "RX_position":[[0, 0, 7],[0, 0, 8],[0, 0, 9],[0, 0, 10]],
            "FOV_azi": [-15, 15],
            "FOV_ele": [-45, 45],
            "DOAFFTSize": 128
        }
        self.device = device

        # Set default parameters for radar hetamap
        self.default = {
            'x_min': -3, 'x_max': 3, 'x_bins': 2000,
            'y_min': 0.4, 'y_max': 4.4, 'y_bins': 1000,
            'z_min': 0, 'z_max': 0.001, 'z_bins': 1
        }

        pass

    def run(self, datas, positions, angles, timestamps, fft_processor):
        """
        Perform Back Projection Algorithm on the input data, GPU is needed to accelerate.

        Parameters:
            datas (np.ndarray): The regular raw radar data.

        Returns:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.
        """

        # Perfome Data Resampling
        datas, positions, angles, timestamps = self.resample_data(datas, positions, angles, timestamps, interval=0.002)

        # Convert all the input data to Tensor
        datas = torch.tensor(datas, dtype=torch.complex64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        angles = torch.tensor(angles, dtype=torch.float, device=self.device)

        # Perform Range FFT
        rangefft_out = fft_processor.range_fft(datas).squeeze()

        # Perform Back Projection Algorithm
        bp_output = self.back_projection(rangefft_out, positions, angles, fov_mask=True)
        pass
    
    def back_projection(self, datas, positions, angles, fov_mask=True):
        """
        Perform Back Projection Algorithm on the input data, GPU is needed to accelerate.

        Parameters:
            datas (torch.Tensor): The output data from Range FFT.
            positions (np.ndarray): The positions of the radar.
            angles (np.ndarray): The angles of the radar.
            timestamps (np.ndarray): The timestamps of the radar.
            
        Returns:
            bp_output (torch.Tensor): The output data from Back Projection Algorithm.
        """

        # Extract basic parameters
        k0 = 2 * torch.pi / self.BPObj["lambda"]

        # Create initial radar heatmap
        grid = self.init_heatmap()
        img = torch.zeros(grid.shape[:-1], dtype=torch.cfloat, device=self.device)
        cnt = torch.ones(grid.shape[:-1], dtype=torch.cfloat, device=self.device)
        img_vec, cnt_vec = img.view(-1), cnt.view(-1)
        
        # Perform Back Projection Algorithm
        for idx in range(datas.shape[0]):
            
            # Extract the position and angle for current frame
            position, angle = positions[idx], angles[idx]
            distances_TXs, distances_RXs = self.cal_distance(grid, position)

            # Loop over all the mimo data in the current frame
            for rx_idx in range(datas.shape[2]):
                for tx_idx in range(datas.shape[3]):
                    
                    distances = (distances_TXs[tx_idx] + distances_RXs[rx_idx]) / 2
                    
                    # 这里分别对应的是索引的上界、下界和小数部分，上面的distance是local的
                    range_idxs_down = (distances / self.BPObj['rangebinSize']).to(torch.uint8)
                    range_idxs_up = (distances / self.BPObj['rangebinSize']).to(torch.uint8) + 1
                    range_idxs_small = (distances / self.BPObj['rangebinSize']) - (distances / self.BPObj['rangebinSize']).to(torch.uint8)

                    # Perform the FOV Mask
                    if fov_mask:
                        FOV_valid_index = self.fov_mask(grid, position, angle, range_idxs_up)
                        print(FOV_valid_index.shape)
                        self.distance_display(FOV_valid_index)
                        aaaa
                        pass

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
        grid = torch.stack((xx, yy, zz), dim=-1).to(dtype=torch.float)
        
        return grid
    
    def resample_data(self, datas, positions, angles, timestamps, interval=0.001):
        """
        Resample the input data in range and time level.

        Parameters:
            datas (np.ndarray): The regular raw radar data.
            positions (np.ndarray): The positions of the radar.
            angles (np.ndarray): The angles of the radar.
            timestamps (np.ndarray): The timestamps of the radar.
        
        Returns:
            resample_datas (np.ndarray): The resampled input data.
            resample_positions (np.ndarray): The resampled positions.
            resample_angles (np.ndarray): The resampled angles.
            resample_timestamps (np.ndarray): The resampled timestamps.
        """

        # Resample the data in chirp level
        datas = datas[:, :, 0:1, :, :]

        # Resample the data in time level
        start_index, end_index = 0, int(self.BPObj["frame_rate"] * (max(timestamps) - min(timestamps) - 1))
        datas = datas[start_index:end_index, :, :, :, :]
        positions = positions[start_index:end_index]
        angles = angles[start_index:end_index]

        # Resample the data in range level
        distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        cumulative_distance, sample_indices = 0, [0]
        for i in range(1, len(distances)):
            cumulative_distance += distances[i - 1]
            if cumulative_distance >= interval:
                sample_indices.append(i)
                cumulative_distance = 0

        # Return the resampled data
        return datas[sample_indices], positions[sample_indices], angles[sample_indices], timestamps[sample_indices]

    def cal_distance(self, grid, position):
        """
        Calculate the distance between the radar and the target grid.

        Parameters:
            grid (torch.Tensor): The heatmap grid.
            position (torch.Tensor): The position of the radar.

        Returns:
            distances_TXs (list): The distances between the radar tx and the target grid.
            distances_RXs (list): The distances between the radar rx and the target grid.
        """

        # Extract the position of the radar
        tx_positions = torch.tensor(self.BPObj["TX_position"], device=self.device) * self.BPObj["antDis"]
        rx_positions = torch.tensor(self.BPObj["RX_position"], device=self.device) * self.BPObj["antDis"]

        # Initialize the position of the radar tx and rx
        chirp_position_TX3 = position + torch.tensor([-0.094, -0.02, 0.032]).to(self.device)
        chirp_tx_positions = list(chirp_position_TX3 + tx_positions[i] for i in [2, 1, 0])
        chirp_rX_positions = list(chirp_position_TX3 + rx_positions[i] for i in [3, 2, 1, 0])

        # Calculate the distance between the radar and the target grid
        def compute_distances(grid, positions):
            distances = []
            for pos in positions:
                distance = torch.sqrt((grid[..., 0] - pos[0])**2 + (grid[..., 1] - pos[1])**2 + (grid[..., 2] - pos[2])**2)
                distances.append(distance)
            return distances
        
        distances_TXs = compute_distances(grid, chirp_tx_positions)
        distances_RXs = compute_distances(grid, chirp_rX_positions)

        # Return the distance
        return distances_TXs, distances_RXs

    def distance_display(self, distances):
        """
        Display the distance between the radar and the target grid.

        Parameters:
            distances (torch.Tensor): The distances between the radar and the target grid.
        """

        # Extract the distance
        distances = distances.squeeze().cpu().numpy()

        # Create the mesh grid
        x = np.arange(0, distances.shape[0])
        y = np.arange(0, distances.shape[1])
        X, Y = np.meshgrid(y, x)

        # Display the 3D mesh of distances
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, distances, cmap='viridis')

        # Set the title and labels
        ax.set_title("3D Mesh of Distances")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Distances")

        plt.show()

    def fov_mask(self, grid, position, angle, range_index):
        """
        Perform Field of View (FOV) Mask on the input data.

        Parameters:
            bp_output (np.ndarray): The output data from Back Projection Algorithm.

        Returns:
            fov_output (np.ndarray): The output data after Field of View (FOV) Mask.
        """

        xx, yy, zz = grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]
        angle_coverted = self.convert_quaternion(angle.cpu().numpy())
        rotation_matrix = torch.tensor(R.from_quat(angle_coverted).as_matrix(), device=self.device, dtype=torch.float32)

        voxels = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()])
        range_idxs_vec = range_index.flatten()
        voxels_transformed = rotation_matrix.T.matmul(voxels - position.reshape(3, 1))

        r = torch.norm(voxels_transformed, dim=0)
        angle_azi = torch.atan2(voxels_transformed[0], voxels_transformed[1]) * 180 / np.pi
        angle_ele = torch.asin(voxels_transformed[2] / r) * 180 / np.pi
        # angle_azi = angle_azi.reshape(grid.shape[:3])
        # self.distance_display(angle_azi)
        # aaaa
        FOV_valid_index = ((angle_azi >= -1 * self.BPObj['FOV_azi'][0]) & (angle_azi <= self.BPObj['FOV_azi'][1]) & (angle_ele >= -1 * self.BPObj['FOV_ele'][0]) & (angle_ele <= self.BPObj['FOV_ele'][1]) & (range_idxs_vec < self.BPObj['DOAFFTSize']))
        
        return FOV_valid_index.reshape(grid.shape[:3])
        return FOV_valid_index

        

    def convert_quaternion(self,q1):
        """
        Converts a quaternion representing a rotation in the x-y-z coordinate system to a quaternion
        representing the same rotation in the x-z-(-y) coordinate system.

        Args:
        q1: (x, y, z, w) quaternion representing a rotation in the x-y-z coordinate system.

        Returns:
        q2: (x, y, z, w) quaternion representing the same rotation in the x-z-(-y) coordinate system.
        """

        # Define a rotation that swaps the y and z axes and inverts the y axis
        swap_yz = R.from_euler('yxz', [0, 90, 0], degrees=True)

        # Convert q1 to a rotation object
        r1 = R.from_quat(q1)

        # Apply the coordinate system transformation and convert back to a quaternion
        r2 = swap_yz * r1 * swap_yz.inv()
        q2 = r2.as_quat()

        return q2

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
        