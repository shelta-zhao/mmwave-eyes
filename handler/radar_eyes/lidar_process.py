"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : This script defines the Lidar Data Processor.
"""

import numpy as np
import open3d as o3d


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

