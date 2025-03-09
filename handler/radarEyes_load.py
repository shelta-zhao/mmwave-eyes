"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Load & Get regular raw radar data of RadarEyes dataset.
"""

import os
import re
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
            return np.load(data_path.replace("ADC", "PCD_SamePaddingUDPERROR", 1))
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


class LidarDataProcessor:

    def save_lidar(self, file_path):
        """
        Read a LiDAR frame, process it by downsampling and denoising, and save the result to a new file.

        Parameters:
            file_path (str): Path to the input LiDAR frame file.

        Note:
            To load the saved data, use the following code:
                cloud_vals = np.fromfile(save_file_name, dtype='float32')
                cloud = cloud_vals.reshape((-1, 4))
        """

        # Read the LiDAR frame
        point_cloud = self.read_lidar_frame(file_path)

        # Downsample and denoise the point cloud
        point_cloud_new = self.downsample_denoise_pointcloud(point_cloud).astype(np.float32)

        # Create the save file name by replacing 'Lidar' with 'Lidar_pcd' in the original file path
        save_file_name = file_path.replace('Lidar', 'Lidar/Lidar_pcd', 1)
        # Save the processed point cloud to the new file
        point_cloud_new.tofile(save_file_name)

        # Use the following code to load the saved data:
        # cloud_vals = np.fromfile(save_file_name,dtype = 'float32')
        # cloud = cloud_vals.reshape((-1,4)

    def cal_vertical(self, byte3, byte4):
        """
        Calculate the vertical angle, sin(vertical angle), and cos(vertical angle)
        from two input bytes.

        Parameters:
            byte3 (int): The first byte.
            byte4 (int): The second byte.

        Returns:
            vertical_angle (float): The calculated vertical angle.
            fSinV_angle (float): The sin(vertical angle).
            fCosV_angle (float): The cos(vertical angle).
        """

        iTempAngle = byte3
        iChannelNumber = iTempAngle >> 6
        iSymmbol = (iTempAngle >> 5) & 0x01

        if iSymmbol == 1:
            iAngle_v = byte4 + byte3 * 256
            fAngle_v = iAngle_v | 0xc000
            if fAngle_v > 32767:
                fAngle_v = fAngle_v - 65536
        else:
            iAngle_Height = iTempAngle & 0x3f
            fAngle_v = byte4 + iAngle_Height * 256

        fAngle_V = fAngle_v * 0.0025

        if iChannelNumber == 0:
            fPutTheMirrorOffAngle = 0
        elif iChannelNumber == 1:
            fPutTheMirrorOffAngle = -2
        elif iChannelNumber == 2:
            fPutTheMirrorOffAngle = -1
        elif iChannelNumber == 3:
            fPutTheMirrorOffAngle = -3
        else:
            fPutTheMirrorOffAngle = -1.4

        fGalvanometrtAngle = fAngle_V + 7.26
        fAngle_R0 = (
            np.cos(30 * np.pi / 180) * np.cos(fPutTheMirrorOffAngle * np.pi / 180) * np.cos(fGalvanometrtAngle * np.pi / 180)
            - np.sin(fGalvanometrtAngle * np.pi / 180) * np.sin(fPutTheMirrorOffAngle * np.pi / 180)
        )
        fSinV_angle = 2 * fAngle_R0 * np.sin(fGalvanometrtAngle * np.pi / 180) + np.sin(fPutTheMirrorOffAngle * np.pi / 180)
        fCosV_angle = np.sqrt(1 - fSinV_angle ** 2)
        vertical_angle = np.arcsin(fSinV_angle) * 180 / np.pi

        return vertical_angle, fSinV_angle, fCosV_angle

    def concat_and_convert_to_signed_int(self, x: int, y: int) -> int:
        """
        Concatenate two integers as hexadecimal values and convert the result
        to a signed integer.

        Parameters:
            x (int): The first integer.
            y (int): The second integer.

        Returns:
            result (int): The signed integer obtained from the concatenation and conversion.
        """

        # Convert x and y to hexadecimal strings
        hex_x = hex(x)[2:].zfill(2)
        hex_y = hex(y)[2:].zfill(2)

        # Concatenate the two hexadecimal strings into one
        hex_result = hex_x + hex_y

        # Convert the concatenated hexadecimal string to a signed integer
        result = int(hex_result, 16)
        if result > 0x7fff:
            result = result - 0x10000

        return result

    def read_lidar_frame(self, lidar_file_name):
        """
        Read and process a LIDAR file and generate a point cloud with instance information.

        Parameters:
            lidar_file_name (str): Path to the LIDAR file.

        Returns:
            colors (np.ndarray): An array of color values associated with each point in the point cloud.
            point_cloud_points (np.ndarray): An array of points (x, y, z, instance) in the point cloud.
        """

        point_cloud_points_string = np.fromfile(lidar_file_name, dtype="|S8")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud_points = []

        for point_index in range(len(point_cloud_points_string)):
            point = point_cloud_points_string[point_index]
            if len(point) != 8:
                continue

            byte1, byte2, byte3, byte4, byte5, byte6, byte7, byte8 = point

            distance = (((byte5 << 8) | byte6) + byte7 / 256) * 0.01 * 25.6

            if (((byte5 << 8) | byte6) > 3):
                instance = byte8
                horizontal = self.concat_and_convert_to_signed_int(byte1, byte2) / 100 - 0.363
                vertical, sin_v, cos_v = self.cal_vertical(byte3, byte4)

                y = distance * cos_v * np.cos(horizontal * np.pi / 180)
                x = distance * cos_v * np.sin(horizontal * np.pi / 180)
                z = distance * sin_v
                point_cloud_points.append([x, y, z, instance])

        point_cloud_points = np.array(point_cloud_points)

        return  point_cloud_points
    
    def downsample_denoise_pointcloud(self, points: np.ndarray, num_points_target: int = 4000, voxel_size: float = 0.02, num_neighbors: int = 20, std_dev_factor: float = 2.0) -> np.ndarray:
        """
        Downsample and denoise a point cloud.

        Parameters:
            points (np.ndarray): Input point cloud as an (N, 4) ndarray.
            num_points_target (int): Target number of downsampled points, default is 4000.
            voxel_size (float): Voxel size, default is 0.02.
            num_neighbors (int): Number of neighboring points, default is 20.
            std_dev_factor (float): Standard deviation multiplier for determining the threshold for removing points, default is 2.0.

        Returns:
            np.ndarray: Downsampled and denoised point cloud as an (M, 4) ndarray, where M <= num_points_target.
        """

        # Convert ndarray to Open3D PointCloud format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Only use the first 3 columns (x, y, z coordinates)

        # Use the 4th column as color information and duplicate it three times to create a pseudo RGB color
        colors = np.tile(points[:, 3].reshape(-1, 1), (1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Downsample using VoxelGrid
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        # Denoise using statistical outlier removal
        denoised_pcd, _ = downsampled_pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_dev_factor)

        # Further downsample if the number of points after downsampling is greater than 4000
        if len(denoised_pcd.points) > num_points_target:
            ratio = num_points_target / len(denoised_pcd.points)
            final_pcd = denoised_pcd.uniform_down_sample(every_k_points=int(1 / ratio))
        else:
            final_pcd = denoised_pcd

        # Convert the downsampled and denoised point cloud back to an ndarray
        final_points = np.hstack((np.asarray(final_pcd.points), np.mean(np.asarray(final_pcd.colors), axis=1).reshape(-1, 1)))
        
        return final_points
