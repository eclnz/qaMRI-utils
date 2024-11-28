import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from skimage.color import gray2rgb
from skimage.measure import block_reduce
import imageio
import os
from numpy.typing import NDArray
from typing import Optional, Tuple, List, Dict
import warnings
import nibabel as nib
from dataclasses import dataclass

from processingUtils import reorient_from_fsl, crop_to_nonzero, apply_crop_bounds
from dcm2bids import display_dropdown_menu, list_bids_subjects_sessions_scans, build_series_list
from userInteraction import prompt_user

plt.switch_backend("Agg")  # For non-GUI environments

@dataclass
class PlotConfig:
    """Configuration for individual MRI plotting."""
    padding: int = 10
    fps: int = 10
    crop: bool = False
    reorient: bool = True
    mask_underlay: bool = False
    mask: Optional[str] = None  # Path or name of the mask image
    underlay_image: Optional[str] = None  # Path or name of the underlay image
    dpi: int = 100
    scale_factor: int = 6
    max_fig_size = 1500

class MRIDataProcessor:
    """Handles preprocessing of MRI data, underlay images, and masks."""
    def __init__(self, 
                 mri_data: NDArray, 
                 config: PlotConfig, 
                 underlay_image: Optional[NDArray] = None, 
                 mask: Optional[NDArray] = None):
        self.mri_data = mri_data
        self.underlay_image = underlay_image
        self.mask = mask
        self.config = config

    def preprocess(self):
        """Reorient data and apply cropping if necessary."""
        
        # find slices
        
        if self.mask is not None:
            self._crop_image()
            if self.config.crop is True:
                # Expand the mask to match the dimensions of self.mri_data
                expanded_mask = self.mask
                while expanded_mask.ndim < self.mri_data.ndim:
                    expanded_mask = expanded_mask[..., np.newaxis]

                # Apply the mask to zero out spatial dimensions
                self.mri_data = self.mri_data * expanded_mask
                
            if self.config.mask_underlay is True and self.underlay_image is not None:
                # Expand the mask to match the dimensions of the underlay image
                expanded_mask = self.mask
                while expanded_mask.ndim < self.underlay_image.ndim:
                    expanded_mask = expanded_mask[..., np.newaxis]

                # Apply the expanded mask to the underlay image
                self.underlay_image = self.underlay_image * expanded_mask

                if self.underlay_image.ndim == 4:
                    # Retain only the first slice along the 4th dimension
                    self.underlay_image = self.underlay_image[:, :, :, 0]
                elif self.underlay_image.ndim == 3:
                    # No changes needed for 3D images
                    pass
                else:
                    raise ValueError(f"Expected 3D or 4D image, but got {self.underlay_image.ndim}D.")
            
        if self.config.reorient:
            self.mri_data = reorient_from_fsl(self.mri_data)
            if self.underlay_image is not None:
                self.underlay_image = reorient_from_fsl(self.underlay_image)
            if self.mask is not None:
                self.mask = reorient_from_fsl(self.mask)

    def _crop_image(self):
        """Applies cropping to the MRI data and mask."""
        cropped_mask, _, crop_bounds = crop_to_nonzero(self.mask, padding=self.config.padding)
        self.mask = cropped_mask
        self.mri_data = apply_crop_bounds(self.mri_data, crop_bounds)

        if self.underlay_image is not None:
            self.underlay_image = apply_crop_bounds(self.underlay_image, crop_bounds)
            
class MRIPlotter:
    """Handles plotting of MRI data in 3D, 4D, or 5D formats."""
    def __init__(self, 
                 mri_data: NDArray, 
                 config: PlotConfig, 
                 output_dir: str,
                 scan_name: str, 
                 underlay_image: Optional[NDArray] = None):
        
        self.mri_data = mri_data
        self.config = config
        self.output_dir = output_dir
        self.scan_name = scan_name
        self.underlay_image = underlay_image
        self.intensity_bounds = (np.min(self.mri_data), np.max(self.mri_data))
        self.video_paths = self._get_video_paths()

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_video_paths(self) -> Dict[str, str]:
        """Generates paths for output videos."""
        return {
            'sagittal': os.path.join(self.output_dir, f"{self.scan_name}_sagittal.mp4"),
            'coronal': os.path.join(self.output_dir, f"{self.scan_name}_coronal.mp4"),
            'axial': os.path.join(self.output_dir, f"{self.scan_name}_axial.mp4")
        }

    def plot(self):
        """Plots the MRI data based on its dimensionality."""
        dimensions = self.mri_data.ndim
        if dimensions == 3:
            self._plot_3d()
        elif dimensions == 4:
            self._plot_4d()
        elif dimensions == 5:
            self._plot_5d()
        else:
            raise ValueError("Unsupported data dimensions. Only 3D, 4D, and 5D data are supported.")
        
    def calculate_compatible_figure_size(self, slice_array: np.ndarray) -> Tuple[float, float]:
        if slice_array.ndim != 2:
            raise ValueError(f"Expected a 2D array for slice dimensions, but got {slice_array.ndim}D array.")
        
        max_size = self.config.max_fig_size

        slice_height, slice_width = slice_array.shape
        macro_block_size = 16  # Default macro block size for compatibility

        # Calculate the raw pixel dimensions
        fig_width_px = slice_width * self.config.scale_factor
        fig_height_px = slice_height * self.config.scale_factor

        # Scale dimensions to ensure the maximum size is adhered to
        scale_ratio = min(max_size / fig_width_px, max_size / fig_height_px)
        fig_width_px *= scale_ratio #type:ignore
        fig_height_px *= scale_ratio #type:ignore

        # Ensure dimensions are compatible with video encoding by rounding up to the nearest macro_block_size
        fig_width_px = int(np.ceil(fig_width_px / macro_block_size) * macro_block_size)
        fig_height_px = int(np.ceil(fig_height_px / macro_block_size) * macro_block_size)

        # Convert to inches using the configured DPI
        fig_width = fig_width_px / self.config.dpi
        fig_height = fig_height_px / self.config.dpi

        return fig_width, fig_height
        
    def extract_three_plane_slices(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts mid-slices for three orthogonal planes (axial, coronal, sagittal) from a 3D volume.

        Parameters:
        - volume: 3D NumPy array representing MRI data.

        Returns:
        - Dict[str, np.ndarray]: A dictionary with keys as plane names and values as extracted slices.
        """
        # Determine mid-slices along each axis
        mid_slices = [dim // 2 for dim in volume.shape]

        # Extract the three orthogonal planes
        planes = {
            "axial": volume[mid_slices[0], :, :], # Remember the dims are no longer x y z because of the reorientation for presentation. It is now z y x 
            "coronal": volume[:, mid_slices[1], :],
            "sagittal": volume[:, :, mid_slices[2]]
        }

        return planes
    
    def add_titles_and_generate_images(
            self, 
            slices: Dict[str, np.ndarray], 
            title_prefix: str = "", 
            intensity_bounds: Optional[Tuple[float, float]] = None
        ) -> Dict[str, np.ndarray]:
        """
        Adds titles and generates images for the provided slices.

        Parameters:
        - slices: Dictionary of plane slices with keys as plane names and values as slice data.
        - title_prefix: Optional title prefix for the slices.
        - intensity_bounds: Tuple specifying (min, max) intensity bounds for consistent scaling.

        Returns:
        - Dict[str, np.ndarray]: A dictionary with keys as plane names and values as images.
        """
        returned_images = {}

        for plane_name, slice_data in slices.items():
            # Use the class method to calculate figure size
            fig_width, fig_height = self.calculate_compatible_figure_size(slice_data)

            # Create the figure
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.config.dpi)
            vmin, vmax = intensity_bounds if intensity_bounds else (None, None)
            ax.imshow(slice_data, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title_prefix} {plane_name.capitalize()} Slice")
            ax.axis("off")
            plt.tight_layout()

            # Render the figure to an image array
            fig.canvas.draw()
            image_array = extract_image_from_canvas(fig)
            returned_images[plane_name] = image_array

            # Close the figure to free memory
            plt.close(fig)

        return returned_images

    def _plot_3d(self):
        """
        Generates images for three orthogonal planes from 3D MRI data and saves them.
        """
        # Generate images for the three planes
        plane_slices = self.extract_three_plane_slices(self.mri_data)
        plane_images = self.add_titles_and_generate_images(plane_slices,title_prefix=self.scan_name, intensity_bounds=self.intensity_bounds)

        # Save the generated images
        for plane, image in plane_images.items():
            output_path = os.path.join(self.output_dir, f"{self.scan_name}_{plane}.png")
            imageio.imwrite(output_path, image)

    def _plot_4d(self):
        """
        Generates videos for three orthogonal planes from 4D MRI data (across timepoints).
        """
        # Initialize video writers for each plane
        video_writers = {
            plane: imageio.get_writer(path, fps=self.config.fps)
            for plane, path in self.video_paths.items()
        }

        # Iterate over timepoints
        for t in range(self.mri_data.shape[-1]):
            # Generate images for the three planes at the current timepoint
            plane_slices = self.extract_three_plane_slices(self.mri_data[..., t])
            plane_images = self.add_titles_and_generate_images(plane_slices,title_prefix=f"{self.scan_name} Time: {t}", 
                                                               intensity_bounds=self.intensity_bounds)

            # Add frames to the respective video writers
            for plane, image in plane_images.items():
                video_writers[plane].append_data(image)

        # Close all video writers
        for writer in video_writers.values():
            writer.close()

    def _plot_5d(self):
        video_writers = {
            plane: imageio.get_writer(path, fps=self.config.fps, codec="libx264")
            for plane, path in self.video_paths.items()
        }
        
        for t in range(self.mri_data.shape[-1]):  # Iterate over timepoints
            
            # Calculate the underlay image slices if it exists
            if self.underlay_image is not None:
                underlay_plane_images = self.extract_three_plane_slices(self.underlay_image)
            
            plane_to_axis = {
                'axial': 0,      # Corresponds to slicing along the Z-axis
                'coronal': 1,   # Corresponds to slicing along the Y-axis
                'sagittal': 2,  # Corresponds to slicing along the X-axis
            }
            
            for plane, axis in plane_to_axis.items():
                
                index = self.mri_data.shape[axis] // 2  # Take the midpoint slice along the axis

                # Extract in-plane displacement components
                if plane == 'sagittal':
                    disp1 = self.mri_data[index, :, :, 1, t]  # Y-component
                    disp2 = self.mri_data[index, :, :, 2, t]  # Z-component
                elif plane == 'coronal':
                    disp1 = self.mri_data[:, index, :, 0, t]  # X-component
                    disp2 = self.mri_data[:, index, :, 2, t]  # Z-component
                elif plane == 'axial':
                    disp1 = self.mri_data[:, :, index, 0, t]  # X-component
                    disp2 = self.mri_data[:, :, index, 1, t]  # Y-component    
                
                if self.underlay_image is not None:    
                    underlay_image = underlay_plane_images[plane]
                else:
                    underlay_image = None

                # Generate the displacement vector frame
                frame = self._plot_displacement_vectors(disp1, disp2, plane, t, underlay_image)
                
                # Write the frame to the video writer
                video_writers[plane].append_data(frame)
        
        for writer in video_writers.values():
            writer.close()

    def _plot_displacement_vectors(self, disp1: NDArray, disp2: NDArray, plane: str, timepoint: int, underlay_image: Optional[NDArray] = None) -> NDArray:
        """
        Plots displacement vectors on a given plane slice at a specific timepoint, with downsampling and color normalization.

        Args:
            disp1 (NDArray): First in-plane displacement component (e.g., X or Y).
            disp2 (NDArray): Second in-plane displacement component (e.g., Y or Z).
            plane (str): Orientation of the plane ('sagittal', 'coronal', 'axial').
            timepoint (int): Timepoint index.

        Returns:
            NDArray: Generated frame as a numpy array (RGB image).
        """
        # Downsample factor for the grid and displacement
        downsample_factor = 5

        # Downsample displacement fields and grid
        X, Y = np.meshgrid(np.arange(disp1.shape[1]), np.arange(disp1.shape[0]))
        X_down = X[::downsample_factor, ::downsample_factor]
        Y_down = Y[::downsample_factor, ::downsample_factor]
        disp1_down = disp1[::downsample_factor, ::downsample_factor]
        disp2_down = disp2[::downsample_factor, ::downsample_factor]

        # Flip the Y-axis for consistent orientation
        Y_down = Y_down[::-1, :]
        
        # Calculate figure dimensions based on slice size and scale by downsampling factor
        fig_width, fig_height = self.calculate_compatible_figure_size(disp1_down)

        # Set up the figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.config.dpi)

        # Compute vector magnitudes for color normalization
        magnitudes = np.sqrt(disp1_down**2 + disp2_down**2)
        vmin, vmax = self.intensity_bounds if self.intensity_bounds else (0, magnitudes.max())

        # Normalize the color based on magnitudes
        norm = Normalize(vmin=0, vmax=vmax)
        sm = ScalarMappable(cmap="viridis", norm=norm)
        colors = sm.to_rgba(magnitudes)

        # Adjust alpha values for zero displacements
        zero_disp_mask = (disp1_down == 0) & (disp2_down == 0)
        colors[:, :, 3] = np.where(zero_disp_mask, 0, colors[:, :, 3])  # Set alpha to 0 for zero vectors

        # Scale the displacement vectors for visualization
        scale_factor = 120
        arrow_thickness = 0.003
        disp1_down_scaled = disp1_down * scale_factor
        disp2_down_scaled = disp2_down * scale_factor
        
        # Plot the underlay if provided
        if underlay_image is not None:
            # Normalize the underlay to [0, 1] and convert to grayscale
            underlay_image = (underlay_image - underlay_image.min()) / (underlay_image.max() - underlay_image.min())
        
        if underlay_image is not None:
            ax.imshow(
                underlay_image,
                cmap="gray",
                interpolation="nearest",
                extent=(0.0, float(disp1.shape[1]), 0.0, float(disp1.shape[0]))
            )

        # Plot the vector field
        Q = ax.quiver(
            X_down, Y_down,
            disp1_down_scaled, disp2_down_scaled,
            angles="xy", scale_units="xy", scale=1, width=arrow_thickness
        )

        # Apply colors to the quiver arrows
        quiver_colors = colors.reshape(-1, 4)
        Q.set_color([tuple(color) for color in quiver_colors])

        # Set title and remove ticks
        ax.set_title(f"{plane.capitalize()} Plane at Timepoint {timepoint}")
        ax.set_xticks([]), ax.set_yticks([])

        plt.tight_layout()  # Ensure optimized layout

        # Render the figure to a frame
        fig.canvas.draw()
        frame = extract_image_from_canvas(fig)

        # Pad the frame to ensure dimensions are divisible by 16
        macro_block_size = 16
        height, width, _ = frame.shape
        padded_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
        padded_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size

        padded_frame = np.zeros((padded_height, padded_width, 3), dtype=frame.dtype)
        padded_frame[:height, :width, :] = frame

        # Close the figure to free memory
        plt.close(fig)

        return frame
    
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

def find_unique_scans_in_bids(bids_folder: str, file_extension: str = ".nii.gz") -> List[str]:
    """
    Scans a BIDS folder and identifies unique scans based on text after `desc-` in filenames.

    Args:
        bids_folder (str): Path to the BIDS folder to scan.
        file_extension (str): File extension to look for (default: '.nii.gz').

    Returns:
        List[str]: A sorted list of unique scan descriptions found in the BIDS folder.
    """

    if not os.path.isdir(bids_folder):
        raise ValueError(f"BIDS folder '{bids_folder}' does not exist or is not a directory.")
    
    subject_session = list_bids_subjects_sessions_scans(bids_folder, file_extension='.nii.gz', raw=False)
    
    unique_scans = build_series_list(subject_session)
    
    return unique_scans

@dataclass
class GroupPlotConfig:
    """Configuration for group plotting."""
    bids_dir: str
    output_dir: str
    selected_scans: list[str]
    all_scans: list[str]

class GroupPlotter:
    """
    Handles plotting for multiple MRI scans with user-configured scan-specific options.
    """

    def __init__(self, config: GroupPlotConfig, subject_session_list: dict[str, dict[str, dict[str, dict[str, str]]]]):
        """
        Initializes the GroupPlotter.

        Args:
            config (GroupPlotConfig): Configuration for the group plotting session.
            subject_session_list (Dict): Dictionary of subjects, sessions, and scans.
        """
        self.config = config
        self.subject_session_list = subject_session_list
        self.scan_configs = self._initialize_scan_configs()

    def _initialize_scan_configs(self) -> dict[str, PlotConfig]:
        """
        Initializes scan options by prompting the user to configure settings for each scan.

        Returns:
            dict: A dictionary mapping scan names to PlotConfig objects.
        """
        configs = {}
        for scan in self.config.selected_scans:
            configs[scan] = self._configure_scan(scan)
        return configs

    def _configure_scan(self, scan: str) -> PlotConfig:
        """
        Prompts the user to configure options for a scan.

        Args:
            scan (str): Scan to configure.

        Returns:
            Dict: Configured options for the scan.
        """
        print(f"\nConfiguring options for scan: {scan}")
        return PlotConfig(
            padding=prompt_user("Enter padding:", default="10", parse_type=int),
            fps=prompt_user("Enter FPS:", default="10", parse_type=int),
            crop=prompt_user("Crop to mask? (y/n):", default="n", parse_type=bool),
            reorient=prompt_user("Reorient data? (y/n):", default="y", parse_type=bool),
            mask=self._select_scan("Select a scan to use as the mask (optional):"),
            underlay_image=self._select_scan("Select a scan to use as the underlay image (optional):"),
            mask_underlay=prompt_user("Mask underlay? (y/n):", default="n", parse_type=bool),
        )
        
    def _find_scan_path(self, subject: str, session: str, scan_name: str|None) -> Optional[str]:
        """
        Finds the path to a scan (e.g., mask or underlay) within the subject-session structure.

        Args:
            subject (str): Subject ID.
            session (str): Session ID.
            scan_name (str): Name of the scan to locate.

        Returns:
            Optional[str]: The path to the scan if found, otherwise None.
        """
        session_data = self.subject_session_list.get(subject, {}).get(session, {})
        for scan, scan_metadata in session_data.items():
            if scan_name and scan_name in scan:
                return scan_metadata["scan_path"]
        return None

    def _select_scan(self, prompt: str) -> Optional[str]:
        """
        Prompts the user to select a scan.

        Args:
            prompt (str): Prompt message.

        Returns:
            Optional[str]: Selected scan or None.
        """
        print("Available scans:")
        for idx, scan in enumerate(self.config.all_scans, 1):
            print(f"{idx}: {scan}")
        while True:
            choice = input(f"{prompt} (enter number or leave blank for none): ").strip()
            if not choice:
                return None
            try:
                index = int(choice) - 1
                if 0 <= index < len(self.config.all_scans):
                    return self.config.all_scans[index]
                print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid input. Enter a number.")

    def plot(self):
        """
        Plots the selected scans using configured options.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        for subject, sessions in self.subject_session_list.items():
            for session, scans in sessions.items():
                for scan, metadata in scans.items():
                    if scan not in self.config.selected_scans:
                        continue

                    try:
                        # Retrieve paths for mask and underlay
                        scan_config = self.scan_configs[scan]
                        mask_path = self._find_scan_path(subject, session, scan_config.mask)
                        underlay_path = self._find_scan_path(subject, session, scan_config.underlay_image)

                        # Load the data
                        mri_data = nib.load(metadata["scan_path"]).get_fdata()
                        mask = nib.load(mask_path).get_fdata() if mask_path else None
                        underlay_image = nib.load(underlay_path).get_fdata() if underlay_path else None

                        # Initialize and preprocess the MRI data
                        processor = MRIDataProcessor(
                            mri_data=mri_data,
                            config=scan_config,
                            underlay_image=underlay_image,
                            mask=mask
                        )
                        processor.preprocess()

                        # Initialize and run the plotter
                        plotter = MRIPlotter(
                            mri_data=processor.mri_data,
                            config=scan_config,
                            output_dir=os.path.join(self.config.output_dir, subject, session),
                            scan_name=scan,
                            underlay_image=processor.underlay_image
                        )
                        plotter.plot()
                        
                        print(f"successfully plotted {scan}")

                    except Exception as e:
                        print(f"Error plotting {scan}: {e}")  
            
# Example usage
if __name__ == "__main__":
    bids_dir = '../qaMRI-clone/testData/BIDS4'
    subject_session_list = list_bids_subjects_sessions_scans(bids_dir, file_extension='.nii.gz', raw = False)
    all_scans = build_series_list(subject_session_list)
    selected_scans = display_dropdown_menu(all_scans, title_text="Select MRI images to plot")
    
    group_config = GroupPlotConfig(bids_dir=bids_dir,
                    output_dir="output_dir",
                    selected_scans=selected_scans,
                    all_scans=all_scans)
     
    plotter = GroupPlotter(group_config, subject_session_list)

    output_directory= 'output'
    # Plot the selected scans
    plotter.plot()