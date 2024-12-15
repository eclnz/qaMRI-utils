import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from typing import Optional, Tuple, List, Dict
import nibabel as nib
from dataclasses import dataclass
from MRISlices import MRISlices
from dcm2bids import display_dropdown_menu, list_bids_subjects_sessions_scans, build_series_list
from userInteraction import prompt_user
from plotConfig import PlotConfig

plt.switch_backend("Agg")  # For non-GUI environments

class MRIDataProcessor:
    """Handles preprocessing of MRI data, underlay images, and masks."""
    def __init__(self, 
                 mri_data_path: str, 
                 config: PlotConfig, 
                 underlay_image_path: Optional[str] = None, 
                 mask_path: Optional[str] = None):
        self.mri_data_path = mri_data_path
        self.underlay_image_path = underlay_image_path
        self.mask_path = mask_path
        self.config = config
        self._import_scans()
        self._extract_slices()

    def _import_scans(self):
        self.mri_data = nib.load(self.mri_data_path)
        self.mask = nib.load(self.mask_path) if self.mask_path else None
        self.underlay_image = nib.load(self.underlay_image_path) if self.underlay_image_path else None
        
    def _extract_slices(self):
        """Extract the middle slices of volumes"""
        # If there is a mask and crop is true, the slices need to be taken from the center of the mask.
        if self.mask is not None:
            if self.config.crop is True: # selected crop
                self.mask_slices = MRISlices.mask_from_nibabel(self.mask)
                mask_locations = self.mask_slices.get_slice_locations()
                mask_cropping = self.mask_slices.get_cropping_bounds()
                self.mri_slices = MRISlices.from_nibabel(self.mri_data, mask_locations, mask_cropping)
                if self.underlay_image is not None:
                    self.underlay_slices = MRISlices.from_nibabel(self.underlay_image, mask_locations, mask_cropping)
                else:
                    self.underlay_slices = None # TODO: Make a default value.
            else:
                self.mask = MRISlices.from_nibabel(self.mask)
                self.mri_slices = MRISlices.from_nibabel(self.mri_data)
                self.underlay_slices = MRISlices.from_nibabel(self.underlay_image)
        else:
            self.mri_slices = MRISlices.from_nibabel(self.mri_data)
            if self.underlay_image is not None:
                self.underlay_slices = MRISlices.from_nibabel(self.underlay_image)
            else: 
                self.underlay_slices = None # TODO: Make a default value.
            
            
class MRIPlotter:
    """Handles plotting of MRI data in 3D, 4D, or 5D formats."""
    def __init__(self, 
                 mri_data: MRISlices, 
                 config: PlotConfig, 
                 output_dir: str,
                 scan_name: str, 
                 underlay_image: Optional[MRISlices] = None):
        
        self.mri_data = mri_data
        self.config = config
        self.output_dir = output_dir
        self.scan_name = scan_name
        self.underlay_image = underlay_image
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
        dimensions = len(self.mri_data.shape)
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
        fig_width_px = slice_width * self.config.figure_scale_factor
        fig_height_px = slice_height * self.config.figure_scale_factor

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

    def _plot_3d(self):
        """
        Generates images for three orthogonal planes from 3D MRI data and saves them.
        """
        # Generate images for the three planes
        plane_images = self.mri_data.add_titles_and_generate_images(config=self.config, title_prefix=self.scan_name)
    
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
            plane_images = self.mri_data.add_titles_and_generate_images(config=self.config, title_prefix=self.scan_name, slice_timepoint=t, underlay_slice=self.underlay_image)

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
        
        # Iterate over timepoints
        for t in range(1, self.mri_data.shape[-1]):
            
            # Generate images for the three planes at the current timepoint
            plane_images = self.mri_data.add_titles_and_generate_images(config=self.config, title_prefix=self.scan_name, slice_timepoint=t, underlay_slice=self.underlay_image)

            # Add frames to the respective video writers
            for plane, image in plane_images.items():
                video_writers[plane].append_data(image)

        # Close all video writers
        for writer in video_writers.values():
            writer.close()

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

                        # Initialize and preprocess the MRI data
                        processor = MRIDataProcessor(
                            mri_data_path=metadata["scan_path"],
                            config=scan_config,
                            underlay_image_path=underlay_path,
                            mask_path=mask_path
                        )

                        # Initialize and run the plotter
                        plotter = MRIPlotter(
                            mri_data=processor.mri_slices,
                            config=scan_config,
                            output_dir=os.path.join(self.config.output_dir, subject, session),
                            scan_name=scan,
                            underlay_image=processor.underlay_slices
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