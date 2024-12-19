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
        config: PlotConfig,
        image: nib.Nifti1Image, 
        mask: Optional[nib.Nifti1Image] = None,
        slice_locations: Optional[Tuple[int, int, int]] = None, 
        cropping_bounds: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
        apply_mask: bool = False
    ) -> "MRISlices":
        # Reorient images to RAS+
        canonical_mri = nib.as_closest_canonical(image)
        shape = canonical_mri.shape
        
        # Load mask file if present and determine crop required
        if mask is not None:
            mask_slices = MRISlices.mask_from_nibabel(mask, config.padding)
            slice_locations = mask_slices.get_slice_locations()
            cropping_bounds = mask_slices.get_cropping_bounds()
            canonical_mask = nib.as_closest_canonical(mask)
            
            if canonical_mri.shape[:3] != canonical_mask.shape[:3]:
                raise ValueError("MRI images and mask are incompatible sizes in the first three dimensions.")

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
        data = canonical_mri.dataobj 

        # Extract slices
        axial = data[axial_slice]
        coronal = data[coronal_slice]
        sagittal = data[sagittal_slice]
        
        if mask is not None and apply_mask:
            mask_data = canonical_mask.dataobj
            mask_axial = mask_data[axial_slice]
            mask_coronal = mask_data[coronal_slice]
            mask_sagittal = mask_data[sagittal_slice]
            
            # Expand dimensions of 2D masks to match 3D data
            while mask_axial.ndim < axial.ndim:
                mask_axial = mask_axial[..., np.newaxis]
            while mask_coronal.ndim < coronal.ndim:
                mask_coronal = mask_coronal[..., np.newaxis]
            while mask_sagittal.ndim < sagittal.ndim:
                mask_sagittal = mask_sagittal[..., np.newaxis]
            
            axial[:,:] *= mask_axial[:,:]
            coronal[:,:] *= mask_coronal[:,:]
            sagittal[:,:] *= mask_sagittal[:,:]
            
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
            affine=canonical_mri.affine,
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
        canonical_mri = nib.as_closest_canonical(image)
        
        # Load full data and crop to non-zero
        data: NDArray = np.asarray(canonical_mri.get_fdata())
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
        axial = cropped_data[new_axial_idx - x_start, :, :]
        coronal = cropped_data[:, new_coronal_idx - y_start, :]
        sagittal =  cropped_data[:, :, new_sagittal_idx - z_start]
        
                # Compute the minimum and maximum intensities across all slices
        intensity_bounds = (
            min(axial.min(), coronal.min(), sagittal.min()),  # Minimum intensity
            max(axial.max(), coronal.max(), sagittal.max())  # Maximum intensity
        )
        
        # Save transform matrix for determining orientation
        affine = canonical_mri.affine

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
        
    def rotate_slices(self)->None:
        '''Rotate the slices for viewing purposes'''
        self.axial = np.flip(np.rot90(self.axial, k=-1, axes=(0, 1)), axis=0)
        self.coronal = np.flip(np.rot90(self.coronal, k=-1, axes=(0, 1)), axis=0)
        self.sagittal =  np.flip(np.rot90(self.sagittal, k=-1, axes=(0, 1)), axis=0)
        
    def add_titles_and_generate_images(
        self,
        config: PlotConfig,
        single_timepoint: bool,
        title_prefix: str = "",
        slice_timepoint: Optional[int] = None,
        underlay_slice: Optional[MRISlices] = None
    ) -> Dict[str, np.ndarray]:
        """
        Adds titles and generates images for the three planes in an MRISlices object.

        Parameters:
            - config (PlotConfig): Configuration for plotting.
            - single_timepoint (bool): Indicates if the data has a single timepoint.
            - title_prefix (str): Optional title prefix for the slices.
            - slice_timepoint (Optional[int]): Timepoint to extract slices from (for multi-timepoint data).
            - underlay_slice (Optional[MRISlices]): Underlay slice for overlays (optional).

        Returns:
            Dict[str, np.ndarray]: A dictionary with plane names ("axial", "coronal", "sagittal") 
            as keys and rendered image arrays as values.
        """
        # Extract planes based on the timepoint
        planes = {
            "axial": self.axial[..., slice_timepoint] if slice_timepoint is not None else self.axial,
            "coronal": self.coronal[..., slice_timepoint] if slice_timepoint is not None else self.coronal,
            "sagittal": self.sagittal[..., slice_timepoint] if slice_timepoint is not None else self.sagittal,
        }

        returned_images = {}
        for plane_name, slice_data in planes.items():
            fig_width, fig_height = self.calculate_compatible_figure_size(slice_data, config=config)

            if slice_data.ndim > 2:
                # Handle displacement images
                underlay_data = getattr(underlay_slice, plane_name) if underlay_slice else None
                disp1, disp2 = extract_in_plane_displacement(slice=slice_data, plane=plane_name)
                slice_data = plot_displacement_vectors(
                    config=config,
                    disp1=disp1,
                    disp2=disp2,
                    fig_size=(fig_width, fig_height),
                    intensity_bounds=self.intensity_bounds,
                    underlay_image=underlay_data,
                )
                scan_title = f"Displacement in {title_prefix} {plane_name.capitalize()} Slice, Timepoint {slice_timepoint}"
            elif not single_timepoint:
                scan_title = f"{title_prefix} {plane_name.capitalize()} Slice, Timepoint {slice_timepoint}"
            else: 
                scan_title = f"{title_prefix} {plane_name.capitalize()} Slice"

            # Plot the slice
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=config.dpi)
            ax.imshow(slice_data, cmap="gray", vmin=self.intensity_bounds[0], vmax=self.intensity_bounds[1])
            ax.set_title(scan_title)
            ax.axis("off")
            plt.tight_layout()

            # Render the figure to an image array
            fig.canvas.draw()
            returned_images[plane_name] = extract_image_from_canvas(fig)
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
    
def plot_displacement_vectors(
    config: PlotConfig, 
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
        
        if underlay_image.ndim == 2:
            display_image = underlay_image  # Already 2D, no changes needed
        elif underlay_image.ndim == 3:
            display_image = underlay_image[:, :, 0]  # Reduce to 2D by taking the first slice along the third dimension
        elif underlay_image.ndim == 4:
            display_image = underlay_image[:, :, 0, 0]  # Reduce to 2D by taking the first slice along the third and fourth dimensions
        else:
            raise ValueError("Input array has unsupported dimensions")
            
        ax.imshow(
            display_image,
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
    # Get the actual size of the canvas
    width, height = fig.canvas.get_width_height()

    # Extract RGBA data from the canvas buffer
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type:ignore

    # Reshape the buffer using dynamically determined width and height
    img_array: NDArray[np.uint8] = buffer.reshape((height, width, 4))  # Shape is (height, width, 4) for RGBA

    # Discard the alpha channel if requested
    if discard_alpha:
        img_array = img_array[:, :, :3]

    return img_array