"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : This script defines all help functions.
"""

import torch
from itertools import count

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

    minVal, maxVal, maxLoc, maxLoc_r = torch.tensor(float('inf'), dtype=torch.float64), torch.tensor(0.0), 0, 0
    absMaxValue, locateMax, initStage, maxData, extendLoc = torch.tensor(0.0), False, True, [], 0

    N = input.shape[0]
    for i in count(0):
        if i >= N + extendLoc - 1:
            break
        i_loc = i % N  
        currentVal = input[i_loc]

        # Update maximum and minimum values
        absMaxValue = torch.max(absMaxValue, currentVal)
        if currentVal > maxVal:
            maxVal = currentVal
            maxLoc = i_loc
            maxLoc_r = i
        
        minVal = torch.min(minVal, currentVal)

        if locateMax:
            if currentVal < (maxVal / gamma):  # Peak found
                maxData.append([maxLoc, maxVal, i - maxLoc_r, maxLoc_r])
                minVal = currentVal
                locateMax = False
        else:
            if currentVal > minVal * gamma:    # Valley found
                locateMax = True
                maxVal = currentVal
                if initStage:
                    extendLoc = i
                    initStage = False

    # Filter peaks based on sidelobe threshold
    absMaxValue_db = absMaxValue * (10 ** (-sidelobeLevel_dB / 10))
    maxData = torch.tensor([data for data in maxData if data[1] >= absMaxValue_db])

    # Convert results to torch tensors
    peakVal, peakLoc = torch.tensor([], dtype=torch.float64), torch.tensor([], dtype=torch.long)
    if len(maxData) > 0:
        peakVal = torch.tensor(maxData[:, 1], dtype=torch.float64)
        peakLoc = torch.tensor(maxData[:, 0] % N, dtype=torch.long)

    return peakVal, peakLoc