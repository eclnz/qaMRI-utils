import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def reorient_from_fsl(mri_data: NDArray, reverse: bool = False, reorient_displacement: bool = False) -> np.ndarray:
    """
    Reorients MRI data from FSL to standard orientation, or optionally in reverse.
    Optionally reorients the displacement direction for 5D data.

    Parameters:
    - mri_data (np.ndarray): A 3D, 4D, or 5D numpy array representing the MRI data.
    - reverse (bool): If True, reverses the reorientation from standard orientation to FSL. Default is False.
    - reorient_displacement (bool): If True, reorients the displacement direction for 5D data. Default is False.

    Returns:
    - np.ndarray: The reoriented (or reversed) MRI data.
    """
    # Validate that the input has 3, 4, or 5 dimensions
    if mri_data.ndim not in {3, 4, 5}:
        raise ValueError(f"Input data must be 3D, 4D, or 5D. Currently has {mri_data.ndim} dimensions.")

    # Disallow displacement reorientation on non-5D data
    if reorient_displacement and mri_data.ndim != 5:
        raise ValueError("Displacement reorientation is only applicable to 5D data. "
                         f"Received {mri_data.ndim}D data with reorient_displacement=True.")

    # Reorient based on the 'reverse' flag
    if not reverse:
        # Standard reorientation from FSL to standard
        if mri_data.ndim == 3:
            mri_reorient = np.transpose(np.flip(np.flip(mri_data, axis=2), axis=1), (2, 1, 0))
        elif mri_data.ndim == 4:
            mri_reorient = np.transpose(np.flip(np.flip(mri_data, axis=2), axis=1), (2, 1, 0, 3))
        elif mri_data.ndim == 5:
            mri_reorient = np.transpose(np.flip(np.flip(mri_data, axis=2), axis=1), (2, 1, 0, 3, 4))
            if reorient_displacement:
                # Reorient displacement directions
                mri_reorient = mri_reorient[:, :, :, [2, 1, 0], :]
                flip_axes = [False, True, True]  # X and Y axes flipped
                for dim in range(3):
                    if flip_axes[dim]:
                        mri_reorient[:, :, :, dim, :] = -mri_reorient[:, :, :, dim, :]
    else:
        # Reverse reorientation from standard orientation back to FSL
        if mri_data.ndim == 3:
            mri_reorient = np.flip(np.flip(np.transpose(mri_data, (2, 1, 0)), axis=1), axis=2)
        elif mri_data.ndim == 4:
            mri_reorient = np.flip(np.flip(np.transpose(mri_data, (2, 1, 0, 3)), axis=1), axis=2)
        elif mri_data.ndim == 5:
            mri_reorient = np.flip(np.flip(np.transpose(mri_data, (2, 1, 0, 3, 4)), axis=1), axis=2)
            if reorient_displacement:
                # Reverse displacement directions
                mri_reorient = mri_reorient[:, :, :, [2, 1, 0], :]
                flip_axes = [False, True, True]  # Y and Z axes flipped
                for dim in range(3):
                    if flip_axes[dim]:
                        mri_reorient[:, :, :, dim, :] = -mri_reorient[:, :, :, dim, :]

    return mri_reorient

def crop_to_nonzero(
    MRIimage: NDArray, 
    padding: int = 0
) -> Tuple[NDArray, NDArray, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Crops the input MRI image to the smallest bounding box containing all non-zero elements, with optional padding.

    Parameters:
    - MRIimage (NDArray[np.float_]): A NumPy array representing the MRI image. Must have at least three spatial dimensions (x, y, z).
    - padding (int): Optional integer specifying the number of voxels to pad around the non-zero elements. Default is 0.

    Returns:
    - cropped_MRIimage (NDArray[np.float_]): The cropped MRI image with dimensions matching the crop bounds.
    - mask (NDArray): A boolean array indicating the non-zero elements within the cropping bounds, including padding.
    - crop_bounds (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]): A 3x2 tuple specifying the start and end indices 
      of the cropping bounds for [x, y, z].

    Raises:
    - ValueError: If MRIimage does not have at least three spatial dimensions.
    - ValueError: If padding is negative.
    """
    if MRIimage.ndim < 3:
        raise ValueError(
            f"Invalid input: Expected a 3D, 4D, or 5D MRI image but got an array with {MRIimage.ndim} dimensions."
        )

    if padding < 0:
        raise ValueError(
            f"Invalid padding: Expected a non-negative integer but got {padding}."
        )

    # Convert MRIimage to a binary array where all values > 0 are set to True
    binary_image = (MRIimage > 0).astype(np.bool_)
    
    if not np.any(binary_image):
        raise ValueError("Empty image: MRIimage contains no non-zero elements, so it cannot be cropped.")

    # Determine crop bounds by finding non-zero elements along each axis
    x_nonzero = np.any(binary_image, axis=(1, 2))
    y_nonzero = np.any(binary_image, axis=(0, 2))
    z_nonzero = np.any(binary_image, axis=(0, 1))
    
    # Find the start and end of non-zero regions along each axis
    try:
        x_start, x_end = np.where(x_nonzero)[0][[0, -1]]
        y_start, y_end = np.where(y_nonzero)[0][[0, -1]]
        z_start, z_end = np.where(z_nonzero)[0][[0, -1]]
    except IndexError as e:
        raise ValueError(
            "Cropping failed: Could not find non-zero bounds along all spatial axes. "
            "Ensure the MRI image has non-zero values across all axes."
        ) from e

    # Apply padding while ensuring bounds stay within the image dimensions
    x_start = max(x_start - padding, 0)
    x_end = min(x_end + padding, MRIimage.shape[0] - 1)
    y_start = max(y_start - padding, 0)
    y_end = min(y_end + padding, MRIimage.shape[1] - 1)
    z_start = max(z_start - padding, 0)
    z_end = min(z_end + padding, MRIimage.shape[2] - 1)

    # Ensure even dimensions for cropping bounds
    if (x_end - x_start + 1) % 2 != 0:
        x_end = min(x_end + 1, MRIimage.shape[0] - 1)
    if (y_end - y_start + 1) % 2 != 0:
        y_end = min(y_end + 1, MRIimage.shape[1] - 1)
    if (z_end - z_start + 1) % 2 != 0:
        z_end = min(z_end + 1, MRIimage.shape[2] - 1)

    # Extract cropping bounds
    crop_bounds = ((x_start, x_end), (y_start, y_end), (z_start, z_end))

    # Crop MRIimage based on calculated bounds
    try:
        cropped_MRIimage = MRIimage[x_start:x_end + 1, y_start:y_end + 1, z_start:z_end + 1]
    except IndexError as e:
        raise ValueError(
            f"Cropping error: The calculated bounds are out of range for the MRI image. "
            f"Bounds: x[{x_start}:{x_end}], y[{y_start}:{y_end}], z[{z_start}:{z_end}]. "
            "Please check the padding and MRI image dimensions."
        ) from e

    # Create the mask array based on the crop bounds
    mask = np.zeros_like(MRIimage, dtype=bool)
    mask[x_start:x_end + 1, y_start:y_end + 1, z_start:z_end + 1] = True

    return cropped_MRIimage, mask, crop_bounds


def apply_crop_bounds(array, crop_bounds):
    """
    Crops the input array according to specified cropping bounds.
    
    Parameters:
    - array: A numpy array to be cropped. Only the first three dimensions are cropped.
    - crop_bounds: A 3x2 array specifying start and end indices for the x, y, and z dimensions.
    
    Returns:
    - cropped_array: The cropped section of the input array.
    
    Raises:
    - ValueError: If crop_bounds does not have the expected 3x2 shape.
    - ValueError: If any of the crop bounds are out of range for the dimensions of the input array.
    """
    # Check crop_bounds shape
    if crop_bounds.shape != (3, 2):
        raise ValueError(
            f"Invalid crop bounds shape: Expected a 3x2 array for crop bounds, but got shape {crop_bounds.shape}."
        )

    x_start, x_end = crop_bounds[0]
    y_start, y_end = crop_bounds[1]
    z_start, z_end = crop_bounds[2]

    # Validate x-axis bounds
    if not (0 <= x_start < array.shape[0]) or not (0 <= x_end < array.shape[0]) or x_start > x_end:
        raise ValueError(
            f"Invalid x-axis bounds: start ({x_start}) and end ({x_end}) must be within [0, {array.shape[0] - 1}] "
            f"and start <= end."
        )

    # Validate y-axis bounds
    if not (0 <= y_start < array.shape[1]) or not (0 <= y_end < array.shape[1]) or y_start > y_end:
        raise ValueError(
            f"Invalid y-axis bounds: start ({y_start}) and end ({y_end}) must be within [0, {array.shape[1] - 1}] "
            f"and start <= end."
        )

    # Validate z-axis bounds
    if not (0 <= z_start < array.shape[2]) or not (0 <= z_end < array.shape[2]) or z_start > z_end:
        raise ValueError(
            f"Invalid z-axis bounds: start ({z_start}) and end ({z_end}) must be within [0, {array.shape[2] - 1}] "
            f"and start <= end."
        )

    # Apply cropping and return the result
    try:
        cropped_array = array[x_start:x_end + 1, y_start:y_end + 1, z_start:z_end + 1]
    except IndexError as e:
        raise ValueError(
            f"Cropping error: Unable to apply crop bounds {crop_bounds} to array of shape {array.shape}. "
            "Please check that crop bounds are within the dimensions of the input array."
        ) from e

    return cropped_array