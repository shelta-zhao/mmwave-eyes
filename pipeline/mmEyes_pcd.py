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
from handler.radarEyes_load import RadarEyesLoader, LidarDataProcessor
from handler.adc_load import get_regular_data
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
            global_point_cloud = []
            radar_ele_all = np.load(os.path.join("data/adc_data", adc_data['prefix'],"1843_ele", "ADC", "all_frames.npy"))
            for frame_idx in tqdm(range(len(synchronized_data['radar_azi']['paths'])), desc="Processing frames", ncols=90):

                # Load data of different sensors
                radar_azi = radarEyesLoader.load_data(synchronized_data['radar_azi']['paths'][frame_idx], sensor='radar_azi')
                lidar = radarEyesLoader.load_data(synchronized_data['lidar']['paths'][frame_idx], sensor="lidar")

                # Perform Distributed Filter


                # Perform Polar Back Projection

                # Perform Cooridaate Transformation
                position = synchronized_data['lidar']['positions'][frame_idx]
                angle = synchronized_data['lidar']['angles'][frame_idx]                
                lidar_transformed = self.transform_point_cloud(lidar, position, angle, transform_flag=("ZED" not in adc_data['camera']))
                global_point_cloud.append(lidar_transformed)

                if frame_idx == 5:
                    self.pcd_display(np.vstack(global_point_cloud))
                    aaaa
                pass
            
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
        transformed_points = rotation.apply(points[:, :3]) + position

        # Restore coordinate system if ZED is not used
        if transform_flag:
            transformed_points[:, [1, 2]] = transformed_points[:, [2, 1]] * [1, -1]

        return transformed_points

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
        # velocity = point_cloud_data[:, 6]

        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color based on frame indices
        # scatter = ax.scatter(x, y, z, c=velocity, cmap='viridis', s=20, alpha=0.8)
        scatter = ax.scatter(x, y, z, s=20, alpha=0.8)

        # Add a color bar
        # cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        # cbar.set_label('Velocity', rotation=270, labelpad=15)
        # scatter.set_clim(-3, 3)

        # Set axis labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')

        # Set title and show
        ax.set_title('3D Point Cloud')
        plt.show()
