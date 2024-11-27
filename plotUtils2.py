import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from skimage.color import gray2rgb
import imageio
import os
from numpy.typing import NDArray
from typing import Optional, Tuple, List, Dict
from processingUtils import reorient_from_fsl, crop_to_nonzero, apply_crop_bounds
from dcm2bids import display_dropdown_menu, list_bids_subjects_sessions_scans
import warnings
import nibabel as nib

plt.switch_backend("Agg")  # For non-GUI environments

class GeneralMRIPlotter:
    """General class for plotting MRI data with adaptation based on dimensionality (3D, 4D, 5D) and optional cropping."""
    
    def __init__(self, mri_data: NDArray, output_dir: str, underlay_image: Optional[NDArray] = None,
                 mask: Optional[NDArray] = None, crop: Optional[bool] = None, padding: int = 10, fps: int = 10,
                 reorient: bool = True, mask_underlay: bool = False):
        """
        Initializes the plotter with MRI data, optional underlay and mask, cropping options, and reorientation.

        Parameters:Can 
        - mri_data: A numpy array representing the MRI data.
        - output_dir: Directory where output files will be saved.
        - underlay_image: Optional underlay image as a numpy array.
        - mask: Optional mask to apply to the MRI data.
        - crop: Optional boolean indicating if cropping should be applied based on the mask. Defaults to True if mask is provided.
        - padding: Padding to add around the cropped area if crop is True.
        - fps: Frames per second for video output.
        - reorient: Boolean indicating if data should be reoriented to standard FSL orientation.
        - mask_underlay: Boolean indicating if the underlay image should be masked by the mask.
        """
        
        # Set default cropping behavior based on the presence of a mask
        self.crop = crop if crop is not None else (mask is not None)

        # Apply reorientation if reorient=True
        self.mri_data = reorient_from_fsl(mri_data) if reorient else mri_data
        self.underlay_image = reorient_from_fsl(underlay_image) if underlay_image is not None and reorient else underlay_image
        self.mask = reorient_from_fsl(mask) if mask is not None and reorient else mask

        self.padding = padding
        self.output_dir = output_dir
        self.fps = fps
        self.mask_underlay = mask_underlay
        os.makedirs(output_dir, exist_ok=True)

        # Validate underlay dimensions if provided
        if self.underlay_image is not None:
            self._validate_underlay_dimensions()
        
        # Apply cropping if crop is True and a mask is provided
        if self.crop and self.mask is not None:
            if self.mask.shape == self.mri_data.shape[:3]:  # Validate mask shape matches spatial dimensions of MRI data
                try:
                    self._apply_cropping()  # Apply cropping to MRI data and underlay image
                    if self.mask_underlay and self.underlay_image is not None:
                        self.underlay_image *= self.mask  # Mask the underlay image if required
                except Exception as e:
                    warnings.warn(
                        f"An error occurred while applying cropping: {e}. "
                        "Cropping was not applied."
                    )
            else:
                warnings.warn(
                    f"Mask shape {self.mask.shape} does not match the spatial dimensions of MRI data {self.mri_data.shape[:3]}. "
                    "Cropping was not applied."
                )

        # Determine and initialize the appropriate plotter based on MRI data dimensionality
        self._select_plotter()

    def _validate_underlay_dimensions(self):
        """Checks if the underlay image has compatible dimensions with the displacement data."""
        if self.underlay_image.ndim == 4:
            self.underlay_image = self.underlay_image[..., 0]
        elif self.underlay_image.ndim != 3:
            raise ValueError(
                f"Underlay image must be 3D or 4D, but got {self.underlay_image.ndim}D."
            )

    def _apply_cropping(self):
        """Applies cropping to mri_data and underlay_image based on non-zero regions in the mask."""
        # Crop mask to find the minimal bounding box and obtain the crop bounds
        cropped_mask, _, crop_bounds = crop_to_nonzero(self.mask, padding=self.padding)
        self.mask = cropped_mask  # Update the mask with the cropped version
        
        # Crop MRI data and underlay image based on the computed crop bounds
        self.mri_data = apply_crop_bounds(self.mri_data, crop_bounds) 
        
        # Apply mask to MRI data with support for 3D, 4D, and 5D MRI data
        if self.mri_data.ndim == 4 and self.mask.ndim == 3:
            # If MRI data is 4D and mask is 3D, expand mask along the last dimension
            self.mri_data *= self.mask[..., np.newaxis]

        elif self.mri_data.ndim == 5 and self.mask.ndim == 3:
            # If MRI data is 5D and mask is 3D, expand mask along the last two dimensions
            self.mri_data *= self.mask[..., np.newaxis, np.newaxis]

        elif self.mri_data.ndim == 5 and self.mask.ndim == 4:
            # If MRI data is 5D and mask is 4D, expand mask along the last dimension
            self.mri_data *= self.mask[..., np.newaxis]

        else:
            # For cases where dimensions match directly (3D mask with 3D MRI, 4D mask with 4D MRI, etc.)
            self.mri_data *= self.mask

        if self.underlay_image is not None:
            self.underlay_image = apply_crop_bounds(self.underlay_image, crop_bounds)

    def _select_plotter(self):
        """Selects the appropriate plotter based on MRI data dimensionality."""
        dimensions = self.mri_data.ndim
        if dimensions == 3:
            self.plotter = MRIPlotter3D(self.mri_data, self.output_dir)
        elif dimensions == 4:
            self.plotter = MRIPlotter4D(self.mri_data, self.output_dir, self.fps)
        elif dimensions == 5:
            self.plotter = MRIPlotter5D(
                self.mri_data, 
                self.output_dir, 
                fps=self.fps, 
                underlay_image=self.underlay_image, 
                mask=self.mask, 
                padding=self.padding
            )
        else:
            raise ValueError("Unsupported MRI data dimensions. Only 3D, 4D, and 5D data are supported.")

    def plot(self):
        """Runs the appropriate plotting method based on data dimensionality."""
        self.plotter.plot()


class MRIPlotter3D:
    """Plots 3D MRI data by displaying or saving slices in three orthogonal planes."""
    
    def __init__(self, mri_data: np.ndarray, output_dir: str, mask: Optional[np.ndarray] = None):
        self.mri_data = self.apply_mask(mri_data, mask)
        self.output_dir = output_dir
        self.intensity_bounds = (np.min(self.mri_data), np.max(self.mri_data))  # Reflects masked data if applied

    def apply_mask(self, mri_data: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Applies a mask to the MRI data if provided, with error handling."""
        if mask is None:
            return mri_data

        try:
            return mri_data * mask
        except (ValueError, TypeError) as e:
            warnings.warn(f"Could not apply mask to MRI data due to: {e}. Proceeding without mask.")
            return mri_data  # Use unmasked data if masking fails

    def plot(self):
        """Plots three orthogonal planes from the 3D MRI volume and saves the image."""
        output_path = os.path.join(self.output_dir, "3D_MRI_slices.png")
        plot_three_planes(self.mri_data, mode="save", output_path=output_path, title_prefix="3D MRI", intensity_bounds=self.intensity_bounds)

class MRIPlotter4D:
    """Plots 4D MRI data by creating separate videos for each plane (axial, coronal, sagittal) across time."""
    
    def __init__(self, mri_data: np.ndarray, output_dir: str, fps: int, mask: Optional[np.ndarray] = None):
        self.mri_data = mri_data * mask[..., np.newaxis] if mask is not None else mri_data
        self.output_dir = output_dir
        self.fps = fps
        self.intensity_bounds = (np.min(mri_data), np.max(mri_data))

    def plot(self):
        """Creates three separate videos for each orthogonal plane across all time points."""
        video_paths = {
            "axial": os.path.join(self.output_dir, "4D_MRI_axial.mp4"),
            "coronal": os.path.join(self.output_dir, "4D_MRI_coronal.mp4"),
            "sagittal": os.path.join(self.output_dir, "4D_MRI_sagittal.mp4")
        }

        video_writers = {
            plane: imageio.get_writer(path, fps=self.fps)
            for plane, path in video_paths.items()
        }

        for t in range(self.mri_data.shape[-1]):
            frames = plot_three_planes(self.mri_data[..., t], mode="return", title_prefix=f"Time {t}", intensity_bounds=self.intensity_bounds)
            for plane, frame in frames.items():
                video_writers[plane].append_data(frame)

        for writer in video_writers.values():
            writer.close()

class MRIPlotter5D:
    """Plots 5D MRI data as a displacement field, displaying vector fields for each time point."""
    
    def __init__(self, displacement_data: np.ndarray, output_dir: str, fps: int = 10, vmin=None, vmax=None,
                 underlay_image: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None, padding: int = 10):
        self.displacement_data:np.ndarray = displacement_data * mask[..., np.newaxis, np.newaxis] if mask is not None else displacement_data
        self.underlay_image:np.ndarray | None = underlay_image if underlay_image is not None else None
        self.output_dir = output_dir
        self.fps = fps
        self.vmin = vmin
        self.vmax = vmax
        self.padding = padding
        self.plotter = DisplacementPlotter()
        self.intensity_max = np.max(displacement_data)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Validate that the spatial dimensions of the underlay image match the displacement data
        if self.underlay_image is not None:
            if self.displacement_data.shape[:3] != self.underlay_image.shape[:3]:
                raise ValueError(
                    f"Mismatch in spatial dimensions: "
                    f"Underlay image dimensions {self.underlay_image.shape[:3]} do not match "
                    f"displacement data dimensions {self.displacement_data.shape[:3]}."
                )


    def plot(self):
        """Creates separate videos for each plane orientation (sagittal, coronal, axial) across time points."""
        # Define output paths for each video
        video_paths = {
            'sagittal': os.path.join(self.output_dir, "5D_MRI_sagittal.mp4"),
            'coronal': os.path.join(self.output_dir, "5D_MRI_coronal.mp4"),
            'axial': os.path.join(self.output_dir, "5D_MRI_axial.mp4")
        }
        
        # Initialize video writers for each orientation
        video_writers = {
            orientation: imageio.get_writer(path, fps=self.fps, codec='libx264')
            for orientation, path in video_paths.items()
        }

        # Define slice indices for each orientation based on the mid-point
        slice_indices = [self.displacement_data.shape[i] // 2 for i in range(3)]
        
        # Loop through each time point to generate frames for each orientation
        for t in range(self.displacement_data.shape[-1]):
            for orientation, index, labels in [
                ('sagittal', slice_indices[2], ["Y", "Z"]),
                ('coronal', slice_indices[1], ["X", "Z"]),
                ('axial', slice_indices[0], ["X", "Y"])
            ]:
                disp1, disp2 = self._get_displacement_components(orientation, index, t)
                underlay_slice = self._get_underlay_slice(orientation, index)
                frame = self.plotter.construct_displacement_vectors(
                    disp1, disp2, f"{orientation.capitalize()} at Time {t}", labels,
                    underlay_image=underlay_slice, vmax=self.intensity_max
                )
                # Write the frame to the respective video writer
                video_writers[orientation].append_data(frame)

        # Close all video writers
        for writer in video_writers.values():
            writer.close()

    def _get_displacement_components(self, orientation: str, index: int, timepoint: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract displacement components for a given orientation, slice index, and timepoint."""
        if orientation == 'sagittal':
            disp1 = self.displacement_data[index, :, :, 1, timepoint]
            disp2 = self.displacement_data[index, :, :, 2, timepoint]
        elif orientation == 'coronal':
            disp1 = self.displacement_data[:, index, :, 0, timepoint]
            disp2 = self.displacement_data[:, index, :, 2, timepoint]
        elif orientation == 'axial':
            disp1 = self.displacement_data[:, :, index, 0, timepoint]
            disp2 = self.displacement_data[:, :, index, 1, timepoint]
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        return disp1, disp2

    def _get_underlay_slice(self, orientation: str, index: int) -> Optional[np.ndarray]:
        """Retrieve an underlay image slice based on the orientation and slice index if available."""
        if self.underlay_image is None:
            return None

        if orientation == 'sagittal':
            return self.underlay_image[index, :, :] # Sagittal (YZ plane): slice along the X-axis
        elif orientation == 'coronal':
            return self.underlay_image[:, index, :] # Coronal (XZ plane): slice along the Y-axis
        elif orientation == 'axial':
            return self.underlay_image[:, :, index] # Axial (XY plane): slice along the Z-axis
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

class DisplacementPlotter:
    """Handles plotting of displacement vectors with optional underlay images."""

    def __init__(self, fig_width: int = 10, fig_height: int = 10, dpi: int = 200):
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.dpi = dpi

    def construct_displacement_vectors(self, disp1: np.ndarray, disp2: np.ndarray, title: str = "",
                                   axis_labels: Optional[List[str]] = None, underlay_image: Optional[np.ndarray] = None,
                                   vmax: Optional[float] = None, arrow_thickness: float = 0.003, scale_factor: int = 120) -> np.ndarray:
        """Plots displacement vectors over an optional underlay image and returns the image as an ndarray."""
        # Calculate figure dimensions based on slice size
        fig_width, fig_height = calculate_compatible_figure_size(disp1)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.dpi)
        downsample_factor = 5
        X, Y = np.meshgrid(np.arange(disp1.shape[1]), np.arange(disp1.shape[0]))
        X_down, Y_down = X[::downsample_factor, ::downsample_factor], Y[::downsample_factor, ::downsample_factor]
        disp1_down, disp2_down = disp1[::downsample_factor, ::downsample_factor], disp2[::downsample_factor, ::downsample_factor]
        Y_down = Y_down[::-1, :]

        if underlay_image is not None:
            underlay_image = self._normalize_underlay(underlay_image)
            ax.imshow(underlay_image, interpolation='nearest', extent=(0.0, float(disp1.shape[1]), 0.0, float(disp1.shape[0])))

        # Plot vector field using quiver
        Q = ax.quiver(X_down, Y_down, disp1_down * scale_factor, disp2_down * scale_factor, 
                    angles='xy', scale_units='xy', scale=1, width=arrow_thickness)

        # Color normalization for vectors based on magnitude
        magnitudes = np.sqrt(disp1_down**2 + disp2_down**2)
        norm = Normalize(vmin=0, vmax=vmax)
        sm = ScalarMappable(cmap="viridis", norm=norm)
        colors = sm.to_rgba(magnitudes)

        # Adjust alpha values for zero displacements
        zero_disp_mask = (disp1_down == 0) & (disp2_down == 0)
        colors[:, :, 3] = np.where(zero_disp_mask, 0, colors[:, :, 3])  # Set alpha to 0 for zero vectors

        # Apply colors to quiver
        quiver_colors = colors.reshape(-1, 4)
        Q.set_color([tuple(color) for color in quiver_colors])

        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])

        plt.tight_layout()  # Ensure optimized layout

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) #type:ignore
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return frame

    def _normalize_underlay(self, underlay_image: np.ndarray) -> np.ndarray:
        """Normalizes a grayscale underlay image and converts it to RGB."""
        underlay_image = (underlay_image - underlay_image.min()) / (underlay_image.max() - underlay_image.min())
        return gray2rgb(underlay_image)

def calculate_divisible_size(size: Tuple[int, int], block_size: int = 16) -> Tuple[int, int]:
    """Ensures width and height are divisible by the specified block size (default is 16)."""
    width, height = size
    width += (block_size - width % block_size) % block_size
    height += (block_size - height % block_size) % block_size
    return width, height

def plot_three_planes(volume: np.ndarray, mode: str = "display", title_prefix: str = "",
                      output_path: Optional[str] = None, intensity_bounds: Optional[Tuple[float, float]] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Plots three orthogonal planes (axial, coronal, sagittal) from a 3D MRI volume,
    with dimensions 5 times larger than the original MRI slice.
    
    Parameters:
    - volume: 3D NumPy array representing MRI data.
    - mode: Mode of operation, either "display", "save", or "return".
        - "display": Show the plot.
        - "save": Save each plane separately with '_<plane>' appended to output_path.
        - "return": Return the images as a dictionary of NumPy arrays.
    - title_prefix: Optional title prefix for the three planes.
    - output_path: Optional path to save the plots as images (used only if mode="save").
    - intensity_bounds: Tuple specifying (min, max) intensity bounds for consistent scaling.

    Returns:
    - If mode="return": Dictionary with keys as plane names ('axial', 'coronal', 'sagittal') and values as images.
    - If mode="save" or mode="display": None.
    """
    mid_slices = [dim // 2 for dim in volume.shape]

    # Set figure dimensions 5 times the size of the MRI slice
    planes = {
        "axial": volume[mid_slices[0], :, :],
        "coronal": volume[:, mid_slices[1], :],
        "sagittal": volume[:, :, mid_slices[2]]
    }

    returned_images = {}

    for plane_name, slice_data in planes.items():
        fig_width, fig_height = calculate_compatible_figure_size(slice_data)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        vmin, vmax = intensity_bounds if intensity_bounds else (None, None)
        ax.imshow(slice_data, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix} {plane_name.capitalize()} Slice")
        ax.axis("off")
        plt.tight_layout()

        if mode == "save" and output_path:
            plane_output_path = f"{output_path}_{plane_name}.png"
            plt.savefig(plane_output_path, dpi=200)
            plt.close(fig)
        elif mode == "return":
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) #type:ignore
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            returned_images[plane_name] = img_array
            plt.close(fig)
        elif mode == "display":
            plt.show()
            plt.close(fig)
        else:
            raise ValueError("Invalid mode. Choose from 'display', 'save', or 'return'.")

    return returned_images if mode == "return" else None

def calculate_compatible_figure_size(slice_array: np.ndarray, scale_factor: float = 5, macro_block_size: int = 16) -> Tuple[float, float]:
    """
    Calculates figure width and height for plotting, scaled from MRI slice dimensions, and ensures dimensions
    are compatible with video encoding by rounding to the nearest multiple of `macro_block_size`.

    Parameters:
    - slice_array: np.ndarray representing the MRI slice, with shape (height, width).
    - scale_factor: Scaling factor to convert slice dimensions to figure size (default 5%).
    - macro_block_size: The size to which dimensions should be rounded for video compatibility (default 16).

    Returns:
    - Tuple[float, float]: Compatible (fig_width, fig_height) in inches.
    """
    if slice_array.ndim != 2:
        raise ValueError(f"Expected a 2D array for slice dimensions, but got {slice_array.ndim}D array.")

    slice_height, slice_width = slice_array.shape

    # Calculate the initial figure dimensions in pixels
    fig_width_px = slice_width * scale_factor
    fig_height_px = slice_height * scale_factor

    # Ensure dimensions are compatible with video encoding by rounding up to the nearest macro_block_size
    fig_width_px = int(np.ceil(fig_width_px / macro_block_size) * macro_block_size)
    fig_height_px = int(np.ceil(fig_height_px / macro_block_size) * macro_block_size)

    # Convert to inches (assuming 100 dpi)
    fig_width = fig_width_px / 100
    fig_height = fig_height_px / 100

    return fig_width, fig_height

def find_unique_scans_in_bids(bids_folder: str, file_extension: str = ".nii.gz") -> List[str]:
    """
    Scans a BIDS folder and identifies unique scans based on text after `desc-` in filenames.

    Args:
        bids_folder (str): Path to the BIDS folder to scan.
        file_extension (str): File extension to look for (default: '.nii.gz').

    Returns:
        List[str]: A sorted list of unique scan descriptions found in the BIDS folder.
    """
    unique_scans = set()

    if not os.path.isdir(bids_folder):
        raise ValueError(f"BIDS folder '{bids_folder}' does not exist or is not a directory.")

    # Recursively traverse the BIDS folder
    for root, _, files in os.walk(bids_folder):
        for file in files:
            if file.endswith(file_extension):
                # Extract the part of the filename after `desc-`
                parts = file.split("_desc-")
                if len(parts) > 1:
                    desc_part = parts[1]  # Get the text after `desc-` and before the next `_`
                    unique_scans.add(desc_part)

    return sorted(unique_scans)

class GroupPlotter:
    """
    Handles plotting for multiple MRI scans with user-defined options for each scan.
    """

    def __init__(self, subject_session_list: Dict, all_scans: List[str], selected_scans: List[str]):
        """
        Initializes the GroupPlotter.

        Args:
            subject_session_list (Dict): A dictionary containing subjects, sessions, and scans.
            selected_scans (List[str]): A list of selected scans to plot.
        """
        self.subject_session_list = subject_session_list
        self.all_scans = all_scans
        self.selected_scans = selected_scans
        self.scan_options = {}

    def collect_scan_options(self, scan: str):
        """
        Collect user-defined plotting options for a single scan.

        Args:
            scan (str): The name of the scan for which to gather options.

        Returns:
            Dict: A dictionary containing the user-specified options for the scan.
        """
        print(f"\nConfiguring options for scan: {scan}")

        # Prompt for underlay image using the dropdown menu
        underlay_image = display_dropdown_menu(
            self.all_scans,
            title_text=f"Select underlay image for scan {scan} (leave blank to skip)"
        )
        underlay_image = underlay_image[0] if underlay_image else None

        # Prompt for mask using the dropdown menu
        mask = display_dropdown_menu(
            self.all_scans,
            title_text=f"Select mask for scan {scan} (leave blank to skip)"
        )
        mask = mask[0] if mask else None

        # Collect other options
        crop = input("Enable cropping? (y/n, default=n): ").strip().lower() or "n" == "y"
        padding = int(input("Padding (default 10): ").strip() or 10)
        fps = int(input("Frames per second for video (default 10): ").strip() or 10)
        reorient = input("Reorient data? (y/n, default=n): ").strip().lower() or "n" == "y"
        mask_underlay = input("Mask underlay image? (y/n, default=n): ").strip().lower() or "n" == "y"

        return {
            "underlay_image": underlay_image,
            "mask": mask,
            "crop": crop,
            "padding": padding,
            "fps": fps,
            "reorient": reorient,
            "mask_underlay": mask_underlay,
        }

    def collect_options(self):
        """
        High-level function to loop through selected scans and collect options for each.
        """
        for scan in self.selected_scans:
            # Call the per-scan option collection method
            self.scan_options[scan] = self.collect_scan_options(scan)

    def plot(self, output_dir: str):
        """
        Plot each selected scan using the user-defined options.

        Args:
            output_dir (str): Directory to save the plotted outputs.
        """
        os.makedirs(output_dir, exist_ok=True)

        for subject, sessions in self.subject_session_list.items():
            for session, scans in sessions.items():
                for scan, metadata in scans.items():
                    if scan not in self.selected_scans:
                        continue  # Skip scans not selected for plotting

                    print(f"\nPlotting scan: {scan} (Subject: {subject}, Session: {session})")
                    options = self.scan_options[scan]

                    # Load data and optional images
                    mri_data = nib.load(metadata["scan_path"]) 
                    mri_data: NDArray = mri_data.get_fdata() # type: ignore
                    underlay_image = np.load(options["underlay_image"]) if options["underlay_image"] else None
                    mask = np.load(options["mask"]) if options["mask"] else None

                    # Initialize and configure the plotter
                    plotter = GeneralMRIPlotter(
                        mri_data=mri_data,
                        output_dir=os.path.join(output_dir, f"{subject}_{session}"),
                        underlay_image=underlay_image,
                        mask=mask,
                        crop=options["crop"],
                        padding=options["padding"],
                        fps=options["fps"],
                        reorient=options["reorient"],
                        mask_underlay=options["mask_underlay"]
                    )
                    plotter.plot()

# Example usage
if __name__ == "__main__":
    subject_session_list = list_bids_subjects_sessions_scans('../qaMRI-clone/testData/BIDS4', file_extension='.nii.gz')
    scans = find_unique_scans_in_bids('../qaMRI-clone/testData/BIDS4')
    selected_scans = display_dropdown_menu(scans, title_text="Select MRI images to plot")

    plotter = GroupPlotter(subject_session_list, scans, selected_scans)
    plotter.collect_options()
    plotter.plot(output_dir="../qaMRI-clone/testData/outputs")



subject_session_list = list_bids_subjects_sessions_scans('../qaMRI-clone/testData/BIDS4', file_extension='.nii.gz')
scans = find_unique_scans_in_bids('../qaMRI-clone/testData/BIDS4')
selected_scans = display_dropdown_menu(scans, title_text="Select MRI images to plot")

print(scans)

for scan in scans:
    print(scan)