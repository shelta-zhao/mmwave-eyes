"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Load & Get regular raw radar data of RadarEyes dataset.
"""

import os
import re
import copy
import open3d as o3d
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

class RadarEyesLoader:

    def __init__(self):
        """
        Initialize the RadarEyesLoader class.
        """

        self.ADC_EXTENSIONS = ['.mat', '.MAT', 'bin', 'BIN', "jpg", "JPG","png","PNG", "npy"]

    def load_data(self, data_path, sensor='radar_azi'):
        """
        Load one frame data from the given path.

        Parameters:
            data_path (str): The path to the data.
            sensor (str): The sensor type to be loaded.
        
        Returns:
            data (np.ndarray): The loaded data from the given path.
        """

        if sensor == 'radar_azi':
            return np.fromfile(data_path.replace("PCD_SamePaddingUDPERROR", "ADC", 1), dtype = "complex128").reshape((1, 128, -1, 4, 3))
        elif sensor == 'radar_ele':
            return np.load(os.path.join(data_path, "1843_ele", "ADC", "all_frames.npy"))
        elif sensor == 'lidar':
            return np.fromfile(data_path.replace("ADC", "Lidar_pcd"), dtype='float32').reshape((-1, 4))
        else:
            raise ValueError(f"Invalid sensor type: {sensor}")

    def data_sync(self, data_path):
        """
        Synchronize the data from different sensors.

        Parameters:
            data_path (str): The path to the data to be synchronized.

        Returns:
            synchronized_data (dict): The synchronized data from different sensors.
            - Sensor data (dict): The data from different sensors.
                - paths (list): The paths to the sensor data.
                - positions (list): The positions of the sensor data.
                - angles (list): The angles of the sensor data.
                - timestamps (list): The timestamps of the sensor data.
        """

        # Get the paths of the data from different sensors
        radar_path_azi = os.path.join(data_path, "1843_azi")
        radar_path_ele = os.path.join(data_path, "1843_ele")
        camera_path = os.path.join(data_path, "Camera_ZED")
        lidar_path = os.path.join(data_path, "Lidar")

        # Initialize the synchronized result
        result = {
            'radar_azi': {'paths': [], 'positions': [], 'angles': [], 'timestamps': []},
            'radar_ele': {'paths': [], 'positions': [], 'angles': [], 'timestamps': []},
            'lidar': {'paths': [], 'positions': [], 'angles': [], 'timestamps': []},
            'camera': {'paths': [], 'positions': [], 'angles': [], 'timestamps': []},
            'metadata': {'start_time': None, 'end_time': None}
        }

        # Read the timestamps from the given paths
        radar_timestamps_azi = self.read_timestamp(radar_path_azi)
        radar_timestamps_ele = self.read_timestamp(radar_path_ele) if os.path.exists(radar_path_ele) else []
        lidar_timestamps = self.read_timestamp(lidar_path)
        camera_timestamps = self.read_timestamp(camera_path)

        # Read the pose data from the given paths
        positions, angles, _ = self.read_pose(camera_path)
 
        # Read the data paths for all sensors
        radar_adc_azi= self.read_data_path(os.path.join(radar_path_azi, 'PCD_SamePaddingUDPERROR'))
        radar_adc_ele = self.read_data_path(os.path.join(radar_path_ele, 'ADC'))
        lidar_adc = self.read_data_path(os.path.join(lidar_path, 'ADC'))
        camera_adc = self.read_data_path(camera_path)

        # Delete the missing frames according to the del_frame.txt
        radar_adc_azi, radar_timestamps_azi = self.del_miss_frame(radar_path_azi, radar_adc_azi, radar_timestamps_azi)
        # if radar_adc_ele:
        #     radar_adc_ele, radar_timestamps_ele = self.del_miss_frame(radar_path_ele, radar_adc_ele, radar_timestamps_ele)

        # Get the start & end time of the data
        start_time = max(map(min, [radar_timestamps_azi, radar_timestamps_ele, lidar_timestamps, camera_timestamps]))
        end_time = min(map(max, [radar_timestamps_azi, radar_timestamps_ele, lidar_timestamps, camera_timestamps]))
        result['metadata'].update({'start_time': start_time, 'end_time': end_time})

        # Perform the data synchronization horizontally
        for idx, radar_time in enumerate(radar_timestamps_azi):
            if start_time < radar_time < end_time:
                # LiDAR Sync
                lidar_idx = np.argmin(np.abs(np.array(lidar_timestamps) - radar_time))

                # Camera Sync
                camera_radar_idx = np.argmin(np.abs(np.array(camera_timestamps) - radar_time))
                camera_lidar_idx = np.argmin(np.abs(np.array(camera_timestamps) - lidar_timestamps[lidar_idx]))
                
                # Save the synchronized data
                result['radar_azi']['paths'].append(radar_adc_azi[idx])
                result['radar_azi']['timestamps'].append(radar_time)
                result['radar_azi']['positions'].append(positions[camera_radar_idx])
                result['radar_azi']['angles'].append(angles[camera_radar_idx])
                
                result['lidar']['paths'].append(lidar_adc[lidar_idx])
                result['lidar']['timestamps'].append(lidar_timestamps[lidar_idx])
                result['lidar']['positions'].append(positions[camera_lidar_idx])
                result['lidar']['angles'].append(angles[camera_lidar_idx])

                result['camera']['paths'].append(camera_path[idx] if camera_adc else "/")
                result['camera']['timestamps'].append(camera_timestamps[camera_radar_idx])
                result['camera']['positions'].append(positions[camera_radar_idx])
                result['camera']['angles'].append(angles[camera_radar_idx])

        # Perform the data synchronization vertically
        if radar_adc_ele:
           
            # Convert to numpy array & filter out the repeated timestamps
            radar_timestamps_ele = np.array(radar_timestamps_ele)
            camera_timestamps = np.array(camera_timestamps)
            positions = np.array(positions)
            angles = np.array(angles)
            positions = positions[np.unique(camera_timestamps, return_index=True)[1]]
            angles = angles[np.unique(camera_timestamps, return_index=True)[1]]
            
            # Create the interpolators for positions and rotations
            positions_interpolator = interp1d(camera_timestamps, positions, axis=0, kind='linear', fill_value='extrapolate')
            rotations = Rotation.from_quat(angles)
            slerp_interpolator = Slerp(camera_timestamps, rotations)

            # Filter out the invalid timestamps
            valid_indices = (radar_timestamps_ele >= camera_timestamps[0]) & (radar_timestamps_ele <= camera_timestamps[-1])
            
            # Save the synchronized data
            # result['radar_ele']['paths'] = np.array(radar_adc_ele)[valid_indices].tolist()
            result['radar_ele']['timestamps'] = radar_timestamps_ele[valid_indices]
            result['radar_ele']['positions'] = positions_interpolator(result['radar_ele']['timestamps'])
            result['radar_ele']['angles'] = slerp_interpolator(result['radar_ele']['timestamps']).as_quat()
        else:
            print("Warning: No elevation radar data found.")

        # Return the synchronized data
        return result
    
    def read_timestamp(self, path):
        """
        Read the timestamps from the given path.
        
        Parameters:
            path (str): The path to the timestamps file.

        Returns:
            timestamps (list): The list of timestamps.
        """

        if os.path.exists(os.path.join(path, 'timestamp.txt')):
            with open(os.path.join(path, 'timestamp.txt')) as f:
                return [float(ts) for ts in f.read().splitlines()[1:-1]]
        else:
            return []

    def read_pose(self, path):
        """
        Read the pose data from the given path.

        Parameters:
            path (str): The path to the pose data.
        
        Returns:
            positions (list): The positions data.
            angles (list): The angles data.
            velocities (list): The velocities data.
        """

        # Read the position & angle data from the given path
        positions, angles, velocities = [], [], []
        with open(os.path.join(path, 'pose.txt')) as f:
            for line in f.read().splitlines()[1:-1]:
                parts = [float(x) for x in line[1:-1].split(",")]
                positions.append([parts[0], -parts[2], parts[1]] if "ZED" not in path else parts[:3])
                angles.append(parts[3:7])
        
        # Read the velocities data if exists
        if os.path.exists('veolcity.txt'):
            with open(os.path.join(path, 'velocity.txt')) as f:
                for line in f.read().splitlines()[1:-1]:
                    parts = [float(x) for x in line[1:-1].split(",")]
                    velocities.append([parts[0], -parts[2], parts[1]])
        
        # Return the pose data
        return positions, angles, velocities
    
    def read_data_path(self, path):
        """
        Read the adc data paths from the given path.

        Parameters:
            path (str): The path to the adc data.

        Returns:
            data_paths (list): The list of adc data paths.
        """

        if not os.path.exists(path):
            return []
        data_paths = [os.path.join(path, f) for f in os.listdir(path) if any(f.endswith(extension) for extension in self.ADC_EXTENSIONS)]
        data_paths.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))

        return data_paths

    def del_miss_frame(self, path, radar_adc, radar_timestamps):
        """
        Delete the missing radar frames from the given path.

        Parameters:
            path (str): The path to the data.
            radar_adc (list): The list of radar adc data.
            radar_timestamps (list): The list of radar timestamps.

        Returns:
            radar_adc (list): The list of radar adc data after deleting the missing frames.
            radar_timestamps (list): The list of radar timestamps after deleting the missing frames.
        """

        try:
            with open(os.path.join(path, "del_frame.txt"), "r") as file:
                del_frame_index = set(int(line.strip()) for line in file.readlines())

            radar_adc = [path for i, path in enumerate(radar_adc) if i not in del_frame_index]
            radar_timestamps = [timestamp for i, timestamp in enumerate(radar_timestamps) if i not in del_frame_index]            
            # print(f"Deleted missing frames : {del_frame_index}")

            if len(radar_adc) != len(radar_timestamps):
                raise Exception("Different number of paths and timestamps after deleting missing frames.")
            
            return radar_adc, radar_timestamps
        except Exception as e:
            print(f"Error: Failed to delete missing frames: {e}")



