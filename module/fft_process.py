"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : Define FFT Processor, including Range FFT & Doppler FFT.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class FFTProcessor:
    def __init__(self, rangeFFTObj, dopplerFFTObj, device='cpu'):
        """
        Initialize the FFTProcessor with range and Doppler FFT configurations.

        Parameters:
            rangeFFTObj (dict): Configuration for range FFT.
            dopplerFFTObj (dict): Configuration for Doppler FFT.
            device (str): The device to perform the computation on ('cpu' or 'cuda').
        """
        
        self.rangeFFTObj = rangeFFTObj
        self.dopplerFFTObj = dopplerFFTObj
        self.device = device

    def run(self, input):
        """
        Perform Range & Doppler FFT on the input data.

        Parameters:
            input (np.ndarray): The input data to be transformed.

        Returns:
            torch.Tensor: The FFT result. Shape: (num_frames, range_fft_size, doppler_fft_size, num_rx, num_tx)
        """

        # Convert input to tensor
        input = torch.tensor(input, dtype=torch.complex64).to(self.device)

        # Perform Range FFT
        range_fft_output = self.range_fft(input)

        # Perform Doppler FFT
        doppler_fft_out = self.doppler_fft(range_fft_output)

        # Return FFT Result
        return doppler_fft_out

    def range_fft(self, input):
        """
        Perform Range FFT on the input data.

        Parameters:
            input (torch.Tensor): The input data to be transformed.

        Returns:
            torch.Tensor: The range FFT result. Shape: (num_frames, range_fft_size, num_chirps, num_rx, num_tx)
        """

        # Get the basic fft parmas
        radar_type, fft_size = self.rangeFFTObj['radarPlatform'], self.rangeFFTObj['rangeFFTSize']
        dc_on, win_on = self.rangeFFTObj['dcOffsetCompEnable'], self.rangeFFTObj['rangeWindowEnable']
        scale_on, scale_factor = self.rangeFFTObj['FFTOutScaleOn'], self.rangeFFTObj['scaleFactorRange']
        discard_on, discardCellLeft, discardCellRight = self.rangeFFTObj['discardEnable'], self.rangeFFTObj['discardCellLeft'], self.rangeFFTObj['discardCellRight']

        # Generate window coefficient
        win_coeff = torch.hann_window(input.shape[1] + 2, periodic=False, dtype=torch.float64).to(self.device)[1:-1]
        # Apply DC offset compensation
        input = input - torch.mean(input, dim=1, keepdim=True) if dc_on else input
        # Apply range-domain windowing
        input = input * win_coeff[:, None, None, None] if win_on else input
        # Perform FFT for each TX/RX chain
        fft_output = torch.fft.fft(input, n=fft_size, dim=1)
        # Apply scale factor
        fft_output = fft_output * scale_factor if scale_on else fft_output
        # Apply range-domain discard
        if discard_on:
            fft_output[:,0:int(fft_size * discardCellLeft),:,:,:] = 0
            fft_output[:,-int(fft_size * discardCellRight):,:,:,:] = 0
        # Phase compensation for IWR6843ISK-ODS
        if radar_type == 'IWR6843ISK-ODS':
            fft_output[:, :, :, 1:3, :] *= -1

        # Return range fft result
        return fft_output

    def doppler_fft(self, input):
        """
        Perform Doppler FFT on the input data (result of range FFT).

        Parameters:
            input (torch.Tensor): The input data to be transformed (result of range fft).

        Returns:
            torch.Tensor: The Doppler FFT result. Shape: (num_frames, range_fft_size, doppler_fft_size, num_rx, num_tx)
        """

        # Get the basic fft parmas
        fft_size, win_on = self.dopplerFFTObj['dopplerFFTSize'], self.dopplerFFTObj['dopplerWindowEnable']
        scale_on, scale_factor = self.dopplerFFTObj['FFTOutScaleOn'], self.dopplerFFTObj['scaleFactorDoppler']

        # Generate window coefficient
        win_coeff = torch.hann_window(input.shape[2] + 2, periodic=True, dtype=torch.float64).to(self.device)[1:-1]
        # Apply Doppler-domain windowing
        input = input * win_coeff[None, None, :, None, None] if win_on else input
        # Perform FFT for each TX/RX chain
        fft_output = torch.fft.fftshift(torch.fft.fft(input, n=fft_size, dim=2), dim=2)
        # Apply scale factor
        fft_output = fft_output * scale_factor if scale_on else fft_output

        # Return doppler fft result
        return fft_output
    
    def fft_display(self, fft_output):
        """
        Plot the 3D Range-Doppler FFT Spectrum.

        Parameters:
        - fft_output: A tensor of shape (range_fft_size, doppler_fft_size).
        """

        # Convert detection results to numpy
        fft_output = fft_output.cpu().numpy()

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
