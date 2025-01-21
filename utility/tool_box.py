"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : This script defines all help functions.
"""





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
