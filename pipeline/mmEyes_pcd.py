"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : mmEyes-PCD pipeline to generate Point Cloud Data (PCD) from raw radar data.
"""

import os
import sys
import yaml
import glob
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from handler.param_process import get_radar_params
from handler.radar_eyes.radarEyes_load import RadarEyesLoader
from handler.radar_eyes.udp_process import UdpDataProcessor
from handler.radar_eyes.lidar_process import LidarDataProcessor
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor
from module.bp_process import BPProcessor


class mmEyesPCD:

    def __init__(self, device='cpu'):

        self.device = device
        
        self.config_path_azi = os.path.join("data/radar_config", "1843_azi")
        self.config_path_ele = os.path.join("data/radar_config", "1843_coherentEle")

    def run(self, adc_list, device, save=False, display=False):
        """
        Perform mmEyes PCD Pipeline.

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
        radarEyesLoader = RadarEyesLoader()
        config_azi = get_radar_params(self.config_path_azi, "AWR1843Boost", load=True)
        config_ele = get_radar_params(self.config_path_ele, "AWR1843Boost", load=True)
        for adc_data in adc_list:
            
            # Print the current data info
            print(f"\nProcessing data: {adc_data['prefix']} | Camera: {adc_data['camera']}")

            # Generate regular data & radar params
            data_path = os.path.join("data/adc_data", f"{adc_data['prefix']}")
            
            # Check if the radar ele data is processed
            if not os.path.exists(os.path.join("data/adc_data", adc_data['prefix'],"1843_ele", "ADC", "all_frames.npy")):
                self.process_radar_ele_data(os.path.join("data/adc_data", adc_data['prefix']))
                print(f"Radar ele data processed successfully: {adc_data['prefix']}")

            # Check if the lidar data is processed
            if not os.path.exists(os.path.join("data/adc_data", adc_data['prefix'],"Lidar/Lidar_pcd")):
                self.process_lidar_data(os.path.join("data/adc_data", adc_data['prefix']))
                print(f"Lidar data processed successfully: {adc_data['prefix']}")

            # Perform data synchronization
            synchronized_data = radarEyesLoader.data_sync(data_path)
            
            # Check if the data is synchronized successfully
            if len(synchronized_data['radar_azi']['paths']) == 0 or len(synchronized_data['lidar']['paths']) == 0:
                print(f"Data synchronization failed. Please check the data : {adc_data['prefix']}.")
                continue

            # Perform mmEyes PCD pipeline for each frame
            global_point_cloud, global_trajectory = [], []
            radar_ele_all = radarEyesLoader.load_data(os.path.join("data/adc_data", adc_data['prefix']))
            for frame_idx in tqdm(range(len(synchronized_data['radar_azi']['paths'])), desc="Processing frames", ncols=90):

                # Load data of different sensors
                timestamp = synchronized_data['radar_azi']['timestamps'][frame_idx]
                radar_ele = self.get_radar_ele(synchronized_data['radar_ele']['timestamps'], timestamp, radar_ele_all)
                radar_azi = radarEyesLoader.load_data(synchronized_data['radar_azi']['paths'][frame_idx], sensor='radar_azi')
                lidar = radarEyesLoader.load_data(synchronized_data['lidar']['paths'][frame_idx], sensor="lidar")

                # Perform Distributed Filter


                # Perform Polar Back Projection

                # Perform Cooridnate Transformation
                angle_radar, position_radar = synchronized_data['radar_azi']['angles'][frame_idx], synchronized_data['radar_azi']['positions'][frame_idx]
                angle_lidar, position_lidar = synchronized_data['lidar']['angles'][frame_idx], synchronized_data['lidar']['positions'][frame_idx]
                transformed_radar = self.transform_point_cloud(radar_azi, position_radar, angle_radar, transform_flag=("ZED" not in adc_data['camera']))
                transformed_lidar = self.transform_point_cloud(lidar, position, angle, transform_flag=("ZED" not in adc_data['camera']))
                
                # Merge the global features
                global_point_cloud.append(transformed_radar)
                global_trajectory.append(position_radar)

                if frame_idx == 10:
                    method_pcd = np.vstack(global_point_cloud)
                    min_height = -1
                    max_height = 3
                    mask = (method_pcd[:, 2] >= min_height-1) & (method_pcd[:, 2] <= max_height+1)
                    method_pcd = method_pcd[mask]
                    method_pcd[:, 3] = method_pcd[:, 3] / np.max(method_pcd[:, 3])
                    self.pcd_display(method_pcd)
                    aaaa
                pass
    
    def process_radar_ele_data(self, data_path):
        """
        Process the raw radar ele data to regular data.

        Parameters:
            data_path (str): The path to the raw radar ele data.
        """

        # Create folders for raw radar ele data and regular data
        udpDataProcessor = UdpDataProcessor()

        # Process raw radar ele data
        udpDataProcessor.save_radar_ele(data_path)

    def process_lidar_data(self, data_path):
        """
        Process the raw lidar data to pcd data.
        
        Parameters:
            data_path (str): The path to the raw lidar data.
        """

        # Create folders for raw lidar data and pcd data
        lidarDataProcessor = LidarDataProcessor()
        src_folder = os.path.join(data_path, "Lidar")
        raw_folder = os.path.join(data_path, "Lidar/ADC")
        pcd_folder = os.path.join(data_path, "Lidar/Lidar_pcd")
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(pcd_folder, exist_ok=True)

        # Process each raw lidar data
        for file_path in tqdm(glob.glob(os.path.join(src_folder, "*.bin")), desc="Processing lidar frames", ncols=90):
            # Process the raw lidar bin file
            lidarDataProcessor.save_lidar(file_path)
            
            # Move the raw bin file to the ADC folder
            shutil.move(file_path, raw_folder)

    def get_radar_ele(self, timestamps, current_timestamp, radar_ele_all, sliding_window=None):
        """
        Get the radar ele data based on the time stamp.

        Parameters:
            timestamps (list): The list of timestamps.
            current_timestamp (int): The time stamp to be searched.
            radar_ele_all (np.ndarray): The radar ele data.
            sliding_window (int): The size of the sliding window.

        Returns:
            radar_ele (np.ndarray): The radar ele data before current timestamp.
        """

        # Get the index of the time stamp
        idx = max(0, np.searchsorted(timestamps, current_timestamp, side='right'))

        # Get the radar ele data
        if sliding_window is not None:
            radar_ele = radar_ele_all[max(0, idx - sliding_window):idx]
        else:
            radar_ele = radar_ele_all[:idx]

        return radar_ele

    def transform_point_cloud(self, points, position, angle, transform_flag):
        """
        Transform the local PCD to global PCD based on the position and angle.

        Parameters:
            points (np.ndarray): The point cloud data to be transformed.
            position (np.ndarray): The position to be transformed.
            angle (np.ndarray): The angle to be transformed.
            transform_flag (str): The flag to determine the transformation type: True if ZED is NOT used.

        Returns:
            transformed_point_cloud (np.ndarray): The transformed point cloud data.
        """
        
        if points.shape[0] == 0:
            return points
        
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(angle)

        # PCD - Camera coordinate transformation
        if transform_flag:
            points[:, [1, 2]] = points[:, [2, 1]] * [1, -1]
            position = np.array([position[0], position[2], -position[1]])

        # Apply rotation and translation
        transformed_xyz = rotation.apply(points[:, :3]) + position

        # Create a new array to hold the transformed points, preserving other features
        transformed_points = np.copy(points)
        transformed_points[:, :3] = transformed_xyz

        # Restore coordinate system if ZED is not used
        if transform_flag:
            transformed_points[:, [1, 2]] = transformed_points[:, [2, 1]] * [1, -1]

        return transformed_points

    def trajectory_display(self, trajectory):
        """
        Display the trajectory.

        Parameters:
            trajectory (np.ndarray): The trajectory to be displayed.
        """

        pass

    def pcd_display(self, point_cloud_data):
        """
        Display the point cloud data.

        Parameters:
            point_cloud_data (np.ndarray): The point cloud data to be displayed.
        """

        # Extract X, Y, Z coordinates
        x = point_cloud_data[:, 0]
        y = point_cloud_data[:, 1]
        z = point_cloud_data[:, 2]
        instance = point_cloud_data[:, 3]
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color based on frame indices
        scatter = ax.scatter(x, y, z, c=instance, cmap='viridis', s=20, alpha=0.8)

        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        # cbar.set_label('Velocity', rotation=270, labelpad=15)
        # scatter.set_clim(-3, 3)

        # Set axis labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')

        # Set title and show
        ax.set_title('3D Point Cloud')
        plt.show()
