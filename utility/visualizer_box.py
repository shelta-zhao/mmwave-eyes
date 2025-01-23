"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : This script converts radar data to Point Cloud Data (PCD) format.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PCD_display(point_cloud_data):
    """
    Plot the point cloud data in a 3D scatter plot.

    Parameters:
    - point_cloud_data: A PyTorch tensor of shape (N, 14), where:
        - Columns 2, 3, 4 are X, Y, Z coordinates.
        - Column 0 is the frame index (optional for coloring).
    """

    # Extract X, Y, Z coordinates
    x = point_cloud_data[:, 2]
    y = point_cloud_data[:, 3]
    z = point_cloud_data[:, 4]
    signal_power = 10 * np.log(10, point_cloud_data[:, 9])

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color based on frame indices
    scatter = ax.scatter(x, y, z, c=signal_power, cmap='viridis', s=20, alpha=0.8)

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Signal Power', rotation=270, labelpad=15)

    # Set axis labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')

    # Set title and show
    ax.set_title('3D Point Cloud')
    plt.show()