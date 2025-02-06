"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : This script converts radar data to Point Cloud Data (PCD) format.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fft_display(fft_output):
    """
    Plot the 3D Range-Doppler FFT Spectrum.

    Parameters:
    - fft_output: A tensor of shape (range_fft_size, doppler_fft_size).
    """

    # Get the magnitude of the FFT output
    x, y = np.arange(fft_output.shape[0]), np.arange(fft_output.shape[1])
    X, Y = np.meshgrid(x, y)

    # Get the magnitude of the FFT output
    Z = np.abs(fft_output.T)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw 3D spectrum
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel("Range FFT Bins")
    ax.set_ylabel("Doppler FFT Bins")
    ax.set_zlabel("Magnitude")

    # Set title and show
    ax.set_title("3D Range-Doppler FFT Spectrum")
    plt.show()


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
    # ax.set_xlim((-2, 2))
    # ax.set_ylim((0, 3))
    # ax.set_zlim((0, 3))

    # Set title and show
    ax.set_title('3D Point Cloud')
    plt.show()