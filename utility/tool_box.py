"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : This script defines all help functions.
"""

import torch


def reshape_fortran(x, shape):
    """
    Reshape a tensor in a Fortran-style (column-major order) while maintaining PyTorch's row-major default.

    Parameters:
        x (torch.Tensor): The input tensor to reshape.
        shape (tuple): The target shape in Fortran-style (column-major order).

    Returns:
        torch.Tensor: The reshaped tensor in the desired shape, maintaining Fortran-style ordering.
    """
    
    if len(x.shape) > 0:
        # Reverse the order of dimensions
        x = x.permute(*reversed(range(len(x.shape))))
    # Reshape and reverse the shape dimensions
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def peak_detect(input, gamma, sidelobeLevel_dB):
    """
    Detect peaks in the input data (input) based on the given gamma factor and sidelobe threshold in dB.

    Parameters:
    - input: 1D array or list of input signal data
    - gamma: Threshold factor for peak detection
    - sidelobeLevel_dB: The minimum required dB difference from the absolute maximum value to consider a peak valid.

    Returns:
    - peakVal: Tensor containing the values of the detected peaks
    - peakLoc: Tensor containing the locations (indices) of the detected peaks
    """

    input = input.flatten()  # Ensure input is a 1D tensor
    N, peaks = input.shape[0], []
    maxVal, maxLoc, absMaxValue, minVal = torch.tensor(0.0), 0, torch.tensor(0.0), torch.tensor(float('inf'))

    for i in range(N):
        currentVal = input[i]

        # Track the absolute maximum value
        absMaxValue = torch.max(absMaxValue, currentVal)

        # Track the current max value and its location
        if currentVal > maxVal:
            maxVal = currentVal
            maxLoc = i

        # Track the minimum value
        minVal = torch.min(minVal, currentVal)

        if currentVal > minVal * gamma:
            if currentVal < maxVal / gamma:
                peaks.append((maxLoc, maxVal, i - maxLoc))  # Store peak info
                minVal = currentVal                         # Update minimum value for next peak detection
                maxVal = currentVal                         # Reset maxVal for next peak search

    # Filter out peaks below sidelobe level
    valid_peaks = []
    absMaxValue_db = absMaxValue * (10 ** (-sidelobeLevel_dB / 10))

    for peak in peaks:
        if peak[1] >= absMaxValue_db:  # Only consider peaks above sidelobe threshold
            valid_peaks.append(peak)

    # Extract peak values and locations
    peakVal = torch.tensor([peak[1] for peak in valid_peaks], dtype=torch.float32)
    peakLoc = torch.tensor([peak[0] for peak in valid_peaks], dtype=torch.long)

    return peakVal, peakLoc

