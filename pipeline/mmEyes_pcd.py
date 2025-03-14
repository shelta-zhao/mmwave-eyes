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
import matplotlib.animation as animation
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
from module.df_process import DFProcessor


class mmEyesPCD:

    def __init__(self, data_root, device='cpu'):

        self.device = device
        self.data_root = data_root
        
        self.config_path_azi = os.path.join("data/radar_config", "1843_azi")
        self.config_path_ele = os.path.join("data/radar_config", "1843_coherentEle")

        self.slider_window = 50

    def run(self, yaml_path, device, save=False, display=False):
        """
        Perform mmEyes PCD Pipeline.

        Parameters:
            yaml_path (str): The list of ADC data to be processed.
            device (str): The device to perform the computation on ('cpu' or 'cuda').
            save (bool): Whether to save the results to a file.
            display (bool): Whether to display the results.

        Returns:
            point_cloud_data (np.ndarray): The generated Point Cloud Data (PCD) from the raw radar data.
        """

        # Parse data config & Get radar params
        with open(f"{yaml_path}.yaml", "r") as file:
            adc_list = yaml.safe_load(file)
        
        # Process each data in the list
        radarEyesLoader = RadarEyesLoader()
        config_azi = get_radar_params(self.config_path_azi, "AWR1843Boost", load=True)
        config_ele = get_radar_params(self.config_path_ele, "AWR1843Boost", load=True)

        # Create all module instances
        bp_processor = BPProcessor(config_ele, device)
        df_processor = DFProcessor(device)

        # for adc_data in adc_list:
        # index = int(yaml_path.split("_")[-1])
        index = 1
        # for adc_data in tqdm(adc_list, desc=f"Processing adc datas {index:02d}", ncols=90, position=index):
        for adc_data in adc_list:
            
            # Print the current data info
            print(f"Processing data: {adc_data['prefix']} | Camera: {adc_data['camera']}")

            # Generate regular data & radar params
            data_path = os.path.join(self.data_root, f"{adc_data['prefix']}")
            
            # Check if the radar ele data is processed
            if not os.path.exists(os.path.join(self.data_root, adc_data['prefix'],"1843_ele", "ADC", "all_frames.npy")):
                self.process_radar_ele_data(os.path.join(self.data_root, adc_data['prefix']))
                print(f"Radar ele data processed successfully: {adc_data['prefix']}")

            # Check if the lidar data is processed
            if not os.path.exists(os.path.join(self.data_root, adc_data['prefix'],"Lidar/Lidar_pcd")):
                self.process_lidar_data(os.path.join(self.data_root, adc_data['prefix']))
                print(f"Lidar data processed successfully: {adc_data['prefix']}")

            # Perform data synchronization
            synchronized_data = radarEyesLoader.data_sync(data_path)
            radar_azi_sync, radar_ele_sync, lidar_sync = synchronized_data['radar_azi'], synchronized_data['radar_ele'], synchronized_data['lidar']
            
            # Check if the data is synchronized successfully
            if len(radar_azi_sync['paths']) == 0 or len(radar_ele_sync['timestamps']) == 0 or len(lidar_sync['paths']) == 0:
                print(f"Data synchronization failed. Please check the data : {adc_data['prefix']}.")
                continue

            # Perform mmEyes PCD pipeline for each frame
            global_radar_pcd, global_lidar_pcd, global_trajectory = [], [], []
            radar_ele_all = radarEyesLoader.load_data(os.path.join(self.data_root, adc_data['prefix']), sensor='radar_ele')[radar_ele_sync['paths']]
            # df_processor.distribute_filter(global_radar_pcd)
            for frame_idx in tqdm(range(len(radar_azi_sync['paths'])), desc="Processing frames", ncols=90):

                # Load data of different sensors                
                radar_azi = radarEyesLoader.load_data(radar_azi_sync['paths'][frame_idx], sensor='radar_azi')
                lidar_pcd = radarEyesLoader.load_data(lidar_sync['paths'][frame_idx], sensor="lidar")

                # Perform Distributed Filter
                radar_azi_pcd = df_processor.run(radar_azi, config_azi)
              
                if frame_idx != 0 and (frame_idx + 1) % self.slider_window == 0:
                    self.pcd_animate(global_radar_pcd, pause=1, BEV=True, save_path="radar_pcd.gif")
                    aaa
                    # tmp = np.vstack(global_radar_pcd)
                    # np.save(f"tmp_radar_pcd_{frame_idx}.npy", tmp)
                    # tmp2 = np.vstack(global_lidar_pcd)
                    # np.save(f"tmp_lidar_pcd_{frame_idx}.npy", tmp2)
                    # asaaa
                    # df_processor.distribute_filter(np.vstack(global_radar_pcd))
                    # print(np.vstack(global_radar_pcd).shape)
                    # print(np.vstack(global_lidar_pcd).shape)
                    # self.pcd_display(np.vstack(global_radar_pcd))
                    # self.pcd_display(np.vstack(global_lidar_pcd))
                    # self.trajectory_display(np.vstack(global_trajectory))
                    # aaaaa

                    # Perform Polar Back Projection
                    start_timestamp, end_timestamp = radar_azi_sync['timestamps'][frame_idx - self.slider_window + 1], radar_azi_sync['timestamps'][frame_idx]
                    radar_ele_datas, radar_ele_positions, radar_ele_angles, radar_ele_timestamps = self.get_radar_ele(start_timestamp, end_timestamp, radar_ele_sync, radar_ele_all)
                    bp_output = bp_processor.run(radar_ele_datas, radar_ele_positions, radar_ele_angles, radar_ele_timestamps)
                    print(bp_output.shape)
                    aaa
                    # Generate the global features
                    pass
                
                # Perfrom PCD Filtering
                # radar_azi_pcd = self.pcd_filter(radar_azi_pcd)
                # lidar_pcd = self.pcd_filter(lidar_pcd)

                # # Perform Cooridnate Transformation
                angle_radar, position_radar = synchronized_data['radar_azi']['angles'][frame_idx], synchronized_data['radar_azi']['positions'][frame_idx]
                angle_lidar, position_lidar = synchronized_data['lidar']['angles'][frame_idx], synchronized_data['lidar']['positions'][frame_idx]
                transformed_radar = self.transform_point_cloud(radar_azi_pcd, position_radar, angle_radar, transform_flag=("ZED" not in adc_data['camera']))
                transformed_lidar = self.transform_point_cloud(lidar_pcd, position_lidar, angle_lidar, transform_flag=("ZED" not in adc_data['camera']))

                # Merge the global features
                global_trajectory.append(radar_azi_sync['positions'][frame_idx])
                global_radar_pcd.append(transformed_radar)
                global_lidar_pcd.append(transformed_lidar)

    
    def process_radar_ele_data(self, data_path):
        """
        Process the raw radar ele data to regular data.

        Parameters:
            data_path (str): The path to the raw radar ele data.
        """

        # Create folders for raw radar ele data and regular data
        udpDataProcessor = UdpDataProcessor()

        # Process raw radar ele data
        try:
            udpDataProcessor.save_radar_ele(data_path)
        except Exception as e:
            print(f"Error processing radar ele data: {data_path}")
            return

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
        try:
            # for file_path in tqdm(glob.glob(os.path.join(src_folder, "*.bin")), desc="Processing lidar frames", ncols=90):
            for file_path in glob.glob(os.path.join(src_folder, "*.bin")):
                # Process the raw lidar bin file
                lidarDataProcessor.save_lidar(file_path)
                
                # Move the raw bin file to the ADC folder
                shutil.move(file_path, raw_folder)
        except Exception as e:
            print(f"Error processing lidar data: {data_path}")
            return

    def get_radar_ele(self, start_timestamp, end_timestamp, radar_ele_sync, radar_ele_all):
        """
        Get the radar ele data based on the time stamp.

        Parameters:
            start_timestamp (float): The start time stamp.
            end_timestamp (float): The end time stamp.
            radar_ele_sync (dict): The synchronized radar ele
            radar_ele_all (np.ndarray): The radar ele data.

        Returns:
            datas (np.ndarray): The radar ele data before current timestamp.
            positions (np.ndarray): The radar ele positions before current timestamp.
            angles (np.ndarray): The radar ele angles before current timestamp.
            timestamps (np.ndarray): The radar ele timestamps before current timestamp.
        """

        # Get the index of the time stamp
        idx_left = max(0, np.searchsorted(radar_ele_sync['timestamps'], start_timestamp, side='left') - 1)
        idx_right = max(0, np.searchsorted(radar_ele_sync['timestamps'], end_timestamp, side='right'))

        # Get the radar ele data
        datas = radar_ele_all[idx_left:idx_right]
        positions = radar_ele_sync['positions'][idx_left:idx_right]
        angles = radar_ele_sync['angles'][idx_left:idx_right]
        timestamps = radar_ele_sync['timestamps'][idx_left:idx_right]

        return datas, positions, angles, timestamps

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

    def pcd_filter(self, point_cloud_data):
        """
        Physically filter the point cloud data.

        Parameters:
            point_cloud_data (np.ndarray): The point cloud data to be filtered. 
        
        Returns:
            filtered_point_cloud (np.ndarray): The filtered point cloud data.
        """

        # Filter the position
        mask = (point_cloud_data[:, 0] >= -3) & (point_cloud_data[:, 0] <= 3) & (point_cloud_data[:, 1] >= 0.4) & (point_cloud_data[:, 1] <= 4) & (point_cloud_data[:, 2] >= -1) & (point_cloud_data[:, 2] <= 3)
        filtered_point_cloud = point_cloud_data[mask]

        # Return the filtered point cloud
        return filtered_point_cloud

    def trajectory_display(self, trajectory, BEV=False):
        """
        Display the trajectory.

        Parameters:
            trajectory (np.ndarray): The trajectory to be displayed.
        """

        # Extract X, Y, Z coordinates
        x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

        # Create 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        ax.plot(x, y, z, color='b', label="Trajectory")

        # Scatter plot for start and end points
        ax.scatter(x[0], y[0], z[0], color='g', marker='o', s=100, label="Start")
        ax.scatter(x[-1], y[-1], z[-1], color='r', marker='x', s=100, label="End")

        # Add labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_ylim(-0.5, 4)
        ax.set_zlim(-1, 3)
        ax.set_title("3D Trajectory Visualization")
        ax.legend()
        
        # Set the view angle
        if BEV:
            ax.view_init(elev=90, azim=-90)
            ax.set_zticks([])

        # Display the plot
        plt.show()

    def pcd_display(self, point_cloud_data, BEV=False):
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

        # Add labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Point Cloud')

        # Set the view angle
        if BEV:
            ax.view_init(elev=90, azim=-90)
            ax.set_zticks([])

        # Display the plot
        plt.show()

    def pcd_animate(self, global_pcd, pause=0.5, BEV=False, save_path=None):
        """
        Display the point cloud data with dynamic updates.

        Parameters:
            point_cloud_data (list of np.ndarray): List of point cloud frames to be displayed sequentially.
            BEV (bool): If True, display in Bird's Eye View (top-down).
            pause_time (float): Pause duration between frames.
        """

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        

        # Define the update function
        def update(frame_idx):
            ax.clear()
            pcd = global_pcd[frame_idx]
            x, y, z, instance = pcd[:, 0], pcd[:, 1], pcd[:, 2], pcd[:, 3]
            scatter = ax.scatter(x, y, z, c=instance, cmap='viridis', s=20, alpha=0.8)
            ax.set_xlim(-8, 8)
            ax.set_ylim(0, 8)
            ax.set_title(f'3D Point Cloud Data : Frame {frame_idx}') 

            if BEV:
                ax.view_init(elev=90, azim=-90)
                ax.set_zticks([])

            return scatter
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(global_pcd), interval=pause * 1000, blit=False, repeat=False)
        
        # Display ans Save the animation
        plt.show()
        if save_path:
            ani.save(save_path, writer='pillow', fps=int(1 / pause))

