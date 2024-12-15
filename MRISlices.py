from __future__ import annotations
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from processingUtils import crop_to_nonzero
from plotConfig import PlotConfig

# TODO: padding needs to add outside to the cropping bounds.
# TODO: zero signal outside of mask if masking

@dataclass
class MRISlices:
    axial: NDArray
    coronal: NDArray
    sagittal: NDArray
    slice_locations: Tuple[int, int, int]
    intensity_bounds: Tuple[int, int]
    affine: NDArray
    cropping_bounds: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None
    shape: Tuple[int, int, int] = field(init=False)

    @classmethod
    def from_nibabel(
        cls, 
        image: nib.Nifti1Image, 
        slice_locations: Optional[Tuple[int, int, int]] = None, 
        cropping_bounds: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None
        ) -> "MRISlices":
        """
        Create an MRISlices object from a nibabel image with optional cropping bounds or slice locations.

        Args:
            image (nib.Nifti1Image): The NIfTI image.
            slice_locations (Optional[Tuple[int, int, int]]): Indices for axial, coronal, and sagittal slices.
                If None, defaults to the middle slices of the first three dimensions of the image.
            cropping_bounds (Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]): Bounds to crop the image 
                before extracting slices.

        Returns:
            MRISlices: An initialized MRISlices object.
        """
        # Reorient image to RAS+
        canonical_image = nib.as_closest_canonical(image)
        
        shape = canonical_image.shape

        if len(shape) < 3:
            raise ValueError("The image must have at least 3 dimensions to extract slices.")

        if slice_locations is None:
            # Default to middle slices of each dimension
            slice_locations = (
                shape[0] // 2,
                shape[1] // 2,
                shape[2] // 2,
            )

        if len(slice_locations) != 3:
            raise ValueError("slice_locations must be a tuple of three integers (axial, coronal, sagittal).")

        # Determine slicing with cropping bounds and slice locations
        slices = [
            slice(bounds[0], bounds[1]) if cropping_bounds and bounds is not None else slice(None)
            for bounds in (cropping_bounds or [(None, None)] * 3)
        ]

        # Add slice locations for axial, coronal, sagittal views
        axial_slice = (slices[0], slices[1], slice_locations[2])  # Axial: (x, y, specific z)
        coronal_slice = (slices[0], slice_locations[1], slices[2])  # Coronal: (x, specific y, z)
        sagittal_slice = (slice_locations[0], slices[1], slices[2])  # Sagittal: (specific x, y, z)
        
        # Access the data and extract slices
        data = canonical_image.dataobj 

        # Extract and rotate slices
        axial = np.rot90(data[axial_slice], k=1, axes=(0, 1))  # Rotate the axial slice
        coronal = np.rot90(data[coronal_slice], k=1, axes=(0, 1))  # Rotate the coronal slice
        sagittal = np.flip(np.rot90(data[sagittal_slice], k=-1, axes=(0, 1)), axis=0)

        
        # Compute the minimum and maximum intensities across all slices
        intensity_bounds = (
            min(axial.min(), coronal.min(), sagittal.min()),  # Minimum intensity
            max(axial.max(), coronal.max(), sagittal.max())  # Maximum intensity
        )

        return cls(
            axial=axial, 
            coronal=coronal, 
            sagittal=sagittal, 
            slice_locations=slice_locations,
            affine=canonical_image.affine,
            cropping_bounds=cropping_bounds,
            intensity_bounds=intensity_bounds
        )

    @classmethod
    def mask_from_nibabel(cls, image: nib.Nifti1Image, padding: int = 0) -> "MRISlices":
        """
        Create an MRISlices object by cropping the image to non-zero elements and calculating slices.

        Args:
            image (nib.Nifti1Image): The NIfTI image to process.
            padding (int): Padding around the non-zero elements. Default is 0.

        Returns:
            MRISlices: An initialized MRISlices object with cropped slices and updated slice locations.
        """
        # Reorient image to RAS+
        canonical_image = nib.as_closest_canonical(image)
        
        # Load full data and crop to non-zero
        data: NDArray = np.asarray(canonical_image.get_fdata())
        cropped_data, _, crop_bounds = crop_to_nonzero(data, padding=padding)

        # Calculate middle slices in the cropped volume
        x_bounds, y_bounds, z_bounds = crop_bounds
        x_start, x_end = x_bounds
        y_start, y_end = y_bounds
        z_start, z_end = z_bounds

        new_axial_idx = (x_start + x_end) // 2
        new_coronal_idx = (y_start + y_end) // 2
        new_sagittal_idx = (z_start + z_end) // 2

        # Extract and rotate slices from the cropped data
        axial = np.rot90(cropped_data[new_axial_idx - x_start, :, :], k=-1, axes=(0, 1))
        coronal = np.rot90(cropped_data[:, new_coronal_idx - y_start, :], k=-1, axes=(0, 1))
        sagittal =  np.flip(np.rot90(cropped_data[:, :, new_sagittal_idx - z_start], k=-1, axes=(0, 1)), axis=0)
        
                # Compute the minimum and maximum intensities across all slices
        intensity_bounds = (
            min(axial.min(), coronal.min(), sagittal.min()),  # Minimum intensity
            max(axial.max(), coronal.max(), sagittal.max())  # Maximum intensity
        )
        
        # Save transform matrix for determining orientation
        affine = canonical_image.affine

        # Return the MRISlices object
        return cls(
            axial=axial,
            coronal=coronal,
            sagittal=sagittal,
            slice_locations=(new_axial_idx, new_coronal_idx, new_sagittal_idx),
            affine=affine,
            intensity_bounds=intensity_bounds,
            cropping_bounds=crop_bounds,
        )

    def __post_init__(self):
        """
        Validate the slices and calculate the shape of the 3D MRI volume.
        """
        # Ensure all slices are at least 2D
        if self.axial.ndim < 2 or self.coronal.ndim < 2 or self.sagittal.ndim < 2:
            raise ValueError(
                "All slices must be at least 2D arrays. "
                f"axial ndim: {self.axial.ndim}, coronal ndim: {self.coronal.ndim}, sagittal ndim: {self.sagittal.ndim}."
            )
            
        # Dynamically calculate the full shape for arbitrary dimensions as (x, y, z, ...)
        self.shape = (
            self.axial.shape[1],   
            self.axial.shape[0],   
            self.sagittal.shape[0],
            *self.axial.shape[2:]  
        )
        
    @staticmethod
    def _apply_cropping(image: nib.Nifti1Image, cropping_bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> nib.Nifti1Image:
        """
        Apply cropping bounds to a NIfTI image.

        Args:
            image (nib.Nifti1Image): The NIfTI image to crop.
            cropping_bounds (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]): Cropping bounds for each axis.

        Returns:
            nib.Nifti1Image: The cropped NIfTI image.
        """
        x_bounds, y_bounds, z_bounds = cropping_bounds
        x_start, x_end = x_bounds
        y_start, y_end = y_bounds
        z_start, z_end = z_bounds

        # Extract the cropped data
        cropped_data = np.asarray(image.dataobj, dtype=float)[x_start:x_end + 1, y_start:y_end + 1, z_start:z_end + 1]

        # Create a new NIfTI image with the cropped data
        cropped_affine = image.affine
        return nib.Nifti1Image(cropped_data, cropped_affine)

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

    def get_cropping_bounds(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """
        Get the cropping bounds used for the image.

        Returns:
            Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]: The cropping bounds.
        """
        return self.cropping_bounds
    
    def apply_mask(self, mask_slices: "MRISlices") -> None:
        """
        Apply the mask from another MRISlices object to the current slices.
        Any elements outside the mask will be zeroed out.

        Args:
            mask_slices (MRISlices): The MRISlices object containing the mask to apply.

        Raises:
            ValueError: If the mask slices do not match the dimensions of the current slices.
        """
        # Validate dimensions for all slices
        if self.axial.shape != mask_slices.axial.shape:
            raise ValueError(f"Axial slice dimensions do not match: {self.axial.shape} vs {mask_slices.axial.shape}")
        if self.coronal.shape != mask_slices.coronal.shape:
            raise ValueError(f"Coronal slice dimensions do not match: {self.coronal.shape} vs {mask_slices.coronal.shape}")
        if self.sagittal.shape != mask_slices.sagittal.shape:
            raise ValueError(f"Sagittal slice dimensions do not match: {self.sagittal.shape} vs {mask_slices.sagittal.shape}")

        # Apply the mask to each slice
        self.axial *= mask_slices.axial
        self.coronal *= mask_slices.coronal
        self.sagittal *= mask_slices.sagittal
        
    def add_titles_and_generate_images(
        self,
        config: PlotConfig,
        title_prefix: str = "",
        slice_timepoint: Optional[int] = None,
        underlay_slice: Optional[MRISlices] = None
    ) -> Dict[str, np.ndarray]:
        """
        Adds titles and generates images for the three planes in an MRISlices object.

        Parameters:
        - mri_slices: An instance of MRISlices containing the three orthogonal slices.
        - title_prefix: Optional title prefix for the slices.
        - intensity_bounds: Tuple specifying (min, max) intensity bounds for consistent scaling.

        Returns:
        - Dict[str, np.ndarray]: A dictionary with keys as plane names ("axial", "coronal", "sagittal")
        and values as rendered image arrays.
        """
        returned_images = {}
        
        # Define specific slices to be used. Handles cases where there are multiple timepoints.
        ## Assumes the timepoint is the last value in the array.
        try:
            if slice_timepoint is not None:
                planes = {
                    "axial": self.axial[..., slice_timepoint],
                    "coronal": self.coronal[..., slice_timepoint],
                    "sagittal": self.sagittal[..., slice_timepoint]
                }
            else:
                planes = {
                    "axial": self.axial,
                    "coronal": self.coronal,
                    "sagittal": self.sagittal
                }
        except IndexError:
            raise ValueError(
                f"Invalid slice_timepoint {slice_timepoint}. Timepoint out of bounds for the provided data."
            )
            
        # Loop through all planes in slices object
        for plane_name, slice_data in planes.items():

            # Correct figure dimensions given configuration settings (dpi,figure_scale_factor, max_fig_size)
            fig_width, fig_height = self.calculate_compatible_figure_size(slice_data, config=config)

            # If the slice data has more than 2 dimensions for a given timepoint, it must be a displacement image. 
            if len(slice_data.shape)>2:
                # Corresponding underlay slice if present
                underlay_data = getattr(underlay_slice, plane_name) if underlay_slice else None
                
                # In plane displacement
                disp1, disp2 = extract_in_plane_displacement(slice=slice_data, plane=plane_name)
                
                # Plot displacement vectors as arrows.
                slice_data = plot_displacement_vectors(
                    config=config, 
                    disp1=disp1, 
                    disp2=disp2,
                    fig_size=(fig_width, fig_height),
                    intensity_bounds=self.intensity_bounds,
                    underlay_image=underlay_data
                )
                
                scan_title = f"Displacement in {title_prefix} {plane_name.capitalize()} Slice, Timepoint {slice_timepoint}"
            else:
                scan_title = f"{title_prefix} {plane_name.capitalize()} Slice, Timepoint {slice_timepoint}"
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=config.dpi)
            vmin, vmax = self.intensity_bounds
            
            ax.imshow(slice_data, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(scan_title)
            ax.axis("off")
            plt.tight_layout()

            # Render the figure to an image array
            fig.canvas.draw()
            image_array = extract_image_from_canvas(fig)
            returned_images[plane_name] = image_array

            # Close the figure to free memory
            plt.close(fig)

        return returned_images

    def calculate_compatible_figure_size(self, slice_array: NDArray, config: PlotConfig) -> Tuple[float, float]:
        """
        Calculate compatible figure size for the given slice data, ensuring dimensions are divisible by 16.

        Args:
        - slice_array (NDArray): A 2D slice array.
        - config (PlotConfig): Configuration for scaling and DPI.

        Returns:
        - Tuple[float, float]: Width and height of the figure in inches, adjusted for macro block size.
        """
        # Extract the first two spatial dimensions
        slice_height, slice_width = slice_array.shape[:2]

        # Calculate the raw pixel dimensions
        fig_width_px = float(slice_width * config.figure_scale_factor)
        fig_height_px = float(slice_height * config.figure_scale_factor)

        # Scale dimensions to ensure the maximum size is adhered to
        scale_ratio = min(config.max_fig_size / fig_width_px, config.max_fig_size / fig_height_px)
        fig_width_px *= scale_ratio
        fig_height_px *= scale_ratio

        # Ensure dimensions are divisible by macro block size (16)
        macro_block_size = 16
        fig_width_px = int((fig_width_px + macro_block_size - 1) // macro_block_size * macro_block_size)
        fig_height_px = int((fig_height_px + macro_block_size - 1) // macro_block_size * macro_block_size)

        # Convert to inches using the default DPI
        fig_width = fig_width_px / config.dpi
        fig_height = fig_height_px / config.dpi

        return fig_width, fig_height
    
def plot_displacement_vectors(config: PlotConfig, 
                              disp1: NDArray, 
                              disp2: NDArray, 
                              fig_size: Tuple[float,float], 
                              intensity_bounds: Tuple[int,int], 
                              underlay_image: Optional[NDArray] = None
                              ) -> NDArray:
    
    # Downsample displacement fields and grid
    X, Y = np.meshgrid(np.arange(disp1.shape[1]), np.arange(disp1.shape[0]))
    X_down = X[::config.displacement_downsample_factor, ::config.displacement_downsample_factor]
    Y_down = Y[::config.displacement_downsample_factor, ::config.displacement_downsample_factor]
    disp1_down = disp1[::config.displacement_downsample_factor, ::config.displacement_downsample_factor]
    disp2_down = disp2[::config.displacement_downsample_factor, ::config.displacement_downsample_factor]

    # Flip the Y-axis for consistent orientation
    Y_down = Y_down[::-1, :]

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=config.dpi)

    # Compute vector magnitudes for color normalization
    magnitudes = np.sqrt(disp1_down**2 + disp2_down**2)
    _, vmax = intensity_bounds

    # Normalize the color based on magnitudes
    norm = Normalize(vmin=0, vmax=vmax)
    sm = ScalarMappable(cmap="viridis", norm=norm)
    colors = sm.to_rgba(magnitudes)

    # Adjust alpha values for zero displacements
    zero_disp_mask = (disp1_down == 0) & (disp2_down == 0)
    colors[:, :, 3] = np.where(zero_disp_mask, 0, colors[:, :, 3])  # Set alpha to 0 for zero vectors

    # Scale the displacement vectors for visualization
    disp1_down_scaled = disp1_down * config.displacement_scale_factor
    disp2_down_scaled = disp2_down * config.displacement_scale_factor
    
    # Plot the underlay if provided
    if underlay_image is not None:
        # Normalize the underlay to [0, 1] and convert to grayscale
        underlay_image = (underlay_image - underlay_image.min()) / (underlay_image.max() - underlay_image.min())
    
    if underlay_image is not None:
        ax.imshow(
            underlay_image[...,0], # Display the first 2 spatial dimensions only., 
            cmap="gray",
            interpolation="nearest",
            extent=(0.0, float(disp1.shape[1]), 0.0, float(disp1.shape[0]))
        )

    # Plot the vector field
    Q = ax.quiver(
        X_down, Y_down,
        disp1_down_scaled, disp2_down_scaled,
        angles="xy", scale_units="xy", scale=1, width=config.displacement_arrow_thickness
    )

    # Apply colors to the quiver arrows
    quiver_colors = colors.reshape(-1, 4)
    Q.set_color([tuple(color) for color in quiver_colors])
    
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()

    # Render the figure to an image array
    fig.canvas.draw()
    image_array = extract_image_from_canvas(fig)
    plt.close(fig)

    return image_array

def extract_in_plane_displacement(slice: NDArray, plane: str) -> Tuple[np.ndarray, np.ndarray]:
        if plane == 'sagittal':
            disp1 = slice[:, :, 1]  # Y-component
            disp2 = slice[:, :, 2]  # Z-component
        elif plane == 'coronal':
            disp1 = slice[:, :, 0]  # X-component
            disp2 = slice[:, :, 2]  # Z-component
        elif plane == 'axial':
            disp1 = slice[:, :, 0]  # X-component
            disp2 = slice[:, :, 1]  # Y-component
        else:
            raise ValueError(f"Invalid plane '{plane}'. Must be one of: 'sagittal', 'coronal', 'axial'.")
        return disp1, disp2

def extract_image_from_canvas(fig: plt.Figure, discard_alpha: bool = True) -> np.ndarray:
    """
    Extracts the rendered image from a Matplotlib figure's canvas as a NumPy array.

    Args:
        fig (plt.Figure): The Matplotlib figure object.
        discard_alpha (bool): Whether to discard the alpha channel (default is True).

    Returns:
        np.ndarray: The image as a NumPy array (RGB or RGBA depending on discard_alpha).
    """
    # Get the actual size of the canvas
    width, height = fig.canvas.get_width_height()

    # Extract RGBA data from the canvas buffer
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type:ignore

    # Reshape the buffer using dynamically determined width and height
    img_array = buffer.reshape((height, width, 4))  # Shape is (height, width, 4) for RGBA

    # Discard the alpha channel if requested
    if discard_alpha:
        img_array = img_array[:, :, :3]

    return img_array