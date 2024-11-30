from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Optional, Tuple
import nibabel as nib
import numpy as np

@dataclass
class MRISlices:
    axial: NDArray
    coronal: NDArray
    sagittal: NDArray
    slice_locations: Tuple[int, int, int]

    @classmethod
    def from_nibabel(cls, image: nib.Nifti1Image, slice_locations: Optional[Tuple[int, int, int]] = None) -> "MRISlices":
        """
        Create an MRISlices object from a nibabel image and optional slice locations.

        Args:
            image (nib.Nifti1Image): The NIfTI image.
            slice_locations (Optional[Tuple[int, int, int]]): Indices for axial, coronal, and sagittal slices.
                If None, defaults to the middle slices of the first three dimensions of the image.

        Returns:
            MRISlices: An initialized MRISlices object.
        """
        # Lazy data loading for efficient slicing
        slicer = image.dataobj

        # Determine slice locations
        if slice_locations is None:
            # Ensure exactly 3 dimensions for middle_slices
            if len(slicer.shape) < 3:
                raise ValueError("The image must have at least 3 dimensions to extract middle slices.")
            slice_locations = (
                int(slicer.shape[0] // 2),
                int(slicer.shape[1] // 2),
                int(slicer.shape[2] // 2),
            )

        # Validate slice_locations
        if len(slice_locations) != 3:
            raise ValueError("slice_locations must be a tuple of three integers (axial, coronal, sagittal).")

        # Extract indices
        axial_idx, coronal_idx, sagittal_idx = slice_locations

        # Extract slices lazily
        axial = np.asarray(slicer[axial_idx, :, :], dtype=np.float32)
        coronal = np.asarray(slicer[:, coronal_idx, :], dtype=np.float32)
        sagittal = np.asarray(slicer[:, :, sagittal_idx], dtype=np.float32)

        # Return an initialized MRISlices object
        return cls(axial=axial, coronal=coronal, sagittal=sagittal, slice_locations=slice_locations)

    def get_slices(self) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Get the axial, coronal, and sagittal slices.

        Returns:
            Tuple[NDArray, NDArray, NDArray]: The extracted slices.
        """
        return self.axial, self.coronal, self.sagittal

    def get_slice_locations(self) -> Tuple[int, int, int]:
        """
        Get the slice locations (indices) used to extract the slices.

        Returns:
            Tuple[int, int, int]: The slice locations (axial, coronal, sagittal).
        """
        return self.slice_locations