"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define Distribute Filtering Algorithm Processor.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from scipy.stats import lognorm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from module.fft_process import FFTProcessor
from module.cfar_process import CFARProcessor
from module.doa_process import DOAProcessor


class DFProcessor:

    def __init__(self, device):

        self.device = device

    def run(self, radar_datas, radar_params):
        
        # Generate all module instances
        fft_processor = FFTProcessor(radar_params['rangeFFTObj'], radar_params['dopplerFFTObj'], self.device)
        cfar_processor = CFARProcessor(radar_params['detectObj'], self.device)
        doa_processor = DOAProcessor(radar_params['DOAObj'], self.device)

        # Perform Range & Doppler FFT
        fft_output = fft_processor.run(radar_datas)
    
        # Perform CFAR-CASO detection
        for frameIdx in range(fft_output.shape[0]):
            detection_results, aa = cfar_processor.run(fft_output[frameIdx,:radar_params['detectObj']['rangeFFTSize'] // 2,:,:,:], frameIdx)

            # Perform DOA Estimation
            doa_results = doa_processor.run(detection_results)
            
            # # Perform Distribute Filtering
            # point_cloud_data = self.distribute_filter(doa_results)

            # Merge the DOA results
            if frameIdx == 0:
                point_cloud_data = doa_results
            else:
                point_cloud_data = np.concatenate((point_cloud_data, doa_results), axis=0)

        # Return the Point Cloud Data
        return point_cloud_data
    
    def distribute_filter(self, point_cloud_data):
        """
        Perform Distribute Filtering on the Point Cloud Data.

        Parameters:
            point_cloud_data (np.ndarray): The Point Cloud Data to be processed.

        Returns:
            np.ndarray: The processed Point Cloud Data.
        """

        point_cloud_data = np.load('tmp_radar_pcd_49.npy')
        # lidar_data = np.load('tmp_lidar_pcd_49.npy')
        # mask = (lidar_data[:, 0] >= -3) & (lidar_data[:, 0] <= 3) & (lidar_data[:, 1] >= 0.4) & (lidar_data[:, 1] <= 4.4) & (lidar_data[:, 2] >= -1) & (lidar_data[:, 2] <= 3)
        # lidar_data = lidar_data[mask]
        # self.pcd_display(lidar_data)
        # aaa
        # self.pcd_display(point_cloud_data)
        # aaaa
        # distributions = ['norm', 'lognorm', 'rayleigh', 'expon', 'weibull_min', 'logistic', 'rice']
        # data = np.log10(1 + point_cloud_data[:, 3]).reshape(-1, 1)
        data = point_cloud_data[:, 3].reshape(-1, 1)    
        gmm = GaussianMixture(n_components=3)
        gmm.fit(data)

        # 获取每个数据点属于每个分布的概率
        probabilities = gmm.predict_proba(data)

        # 获取每个数据点最有可能的分布（即概率最大的分布）
        labels = np.argmax(probabilities, axis=1)

        # 输出拟合结果
        print("GMM means:", gmm.means_)
        print("GMM covariances:", gmm.covariances_)
        print("GMM weights:", gmm.weights_)

        # Extract X, Y, Z coordinates
        x = -point_cloud_data[:, 0]
        y = point_cloud_data[:, 1]
        z = point_cloud_data[:, 2]

        selected_label = [1, 2]
        print(selected_label)
        x_filtered = x[np.isin(labels, selected_label)]
        y_filtered = y[np.isin(labels, selected_label)]
        z_filtered = z[np.isin(labels, selected_label)]
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color based on frame indices
        scatter = ax.scatter(x, y, z, c=labels, cmap='viridis', s=20, alpha=0.8)
        # scatter = ax.scatter(x_filtered, y_filtered, z_filtered,  cmap='viridis', s=20, alpha=0.8)

        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        # cbar.set_label('Velocity', rotation=270, labelpad=15)
        # scatter.set_clim(-3, 3)

        # Add labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Point Cloud')

        # Display the plot
        plt.show()
        aaaa
        # # 可视化每个点所属的分布
        # plt.scatter(range(len(data)), data, c=labels, cmap='viridis')
        # plt.xlabel("Data points")
        # plt.ylabel("Values")
        # plt.title("Point Cloud Data Distribution Assignment")
        # plt.show()

        # aaaa
        distributions = ['norm', 'lognorm', 'rayleigh', 'expon', 'weibull_min', 'logistic', 'rice']
        point_cloud_data = np.load('tmp_radar_pcd_49.npy')
        data = np.log10(1 + point_cloud_data[:, 3])
        
        results = {}
        print("start to fit data")
        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            D, p_value = stats.kstest(data, dist_name, args=params)
            results[dist_name] = {
                'params': params,
                'D': D,
                'p_value': p_value
            }
        
        for dist_name in distributions:
            print(f"{dist_name}: {results[dist_name]['D']} {results[dist_name]['p_value']}")

        
        x = np.linspace(min(data), max(data), 1000)

        plt.figure(figsize=(10, 6))

        # 遍历所有分布并绘制拟合曲线
        for dist_name, result in results.items():
            dist = getattr(stats, dist_name)
            params = result['params']
            
            # 对于 lognorm 分布，确保传递正确的参数
            if dist_name == 'lognorm':
                # lognorm 分布的参数是 (s, loc, scale)，因此要确保按正确顺序传递参数
                pdf = dist.pdf(x, params[0], loc=params[1], scale=params[2])
            else:
                # 对于其他分布，直接解包params
                pdf = dist.pdf(x, *params)
            
            plt.plot(x, pdf, label=dist_name)

        # 设置图形的标题和标签
        plt.title("Fitted Distributions")
        plt.xlabel("Data Values")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)

        # 显示图形
        plt.show()
        aaaa
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
        instance = np.log10(1 + point_cloud_data[:, 3])
        
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

        # Display the plot
        plt.show()