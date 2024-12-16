import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from typing import Optional, Tuple, List, Dict
import nibabel as nib
from dataclasses import dataclass
from MRISlices import MRISlices
from MRIVideo import MRIMedia
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
        self._get_media_type()

    def _import_scans(self):
        self.mri_data = nib.load(self.mri_data_path)
        self.mask = nib.load(self.mask_path) if self.mask_path else None
        self.underlay_image = nib.load(self.underlay_image_path) if self.underlay_image_path else None
        
    def _extract_slices(self):
        """Extract the middle slices of volumes"""
        
        # Check if the underlay matches spatial dimensions if present.
        if self.underlay_image is not None:
            if self.underlay_image.shape[:3] != self.mri_data.shape[:3]:
                raise ValueError("Size of underlay image does not match size of MRI data in the first three dimensions")
                
        # If mask present
        if self.mask is not None:
            
            # Check if the mask matches spatial dimensions if present.
            if self.mask.shape[:3] != self.mri_data.shape[:3]:
                raise ValueError("Size of mask does not match size of MRI data in the first three dimensions")
            
            # User specifies to crop the image
            if self.config.crop is True:
                
                # Read MRI with the mask applied
                self.mri_slices = MRISlices.from_nibabel(self.config,self.mri_data, self.mask)
                
                # If the user also supplies underlay
                if self.underlay_image is not None:
                    
                    # Read underlay with mask applied 
                    # TODO:always applies mask to underlay, while this should be specified by user.
                    self.underlay_slices = MRISlices.from_nibabel(self.config,self.underlay_image, self.mask)
                # If user does not supply underlay its set to None
                else:
                    self.underlay_slices = None
            # If user provides mask but does not set crop to true
            #TODO: if mask is not provided then it does not get applied. I need a new method which applies the masking seperately to be called here.
            else:
                self.mri_slices = MRISlices.from_nibabel(self.config,self.mri_data)
                self.underlay_slices = MRISlices.from_nibabel(self.config,self.underlay_image)
        # If user does not provide a mask
        else:
            # Read mri image normally
            self.mri_slices = MRISlices.from_nibabel(self.config,self.mri_data)
            
            # If user provides underlay, then read that in without masking
            if self.underlay_image is not None:
                self.underlay_slices = MRISlices.from_nibabel(self.config,self.underlay_image)
            # If no underlay supplied its set to none.
            else: 
                self.underlay_slices = None
                
        # Make sure the images are rotated from RSL so they appear normal clinically
        self.mri_slices.rotate_slices()    
        if self.underlay_image is not None:
            self.underlay_slices.rotate_slices()    
                
    def _get_media_type(self)-> None:
        slice_dims = len(self.mri_slices.axial.shape)
        if slice_dims == 2:
            self.media_type='png'
        elif slice_dims > 2:
            self.media_type='mp4'
            
class MRIPlotter:
    """Handles plotting of MRI data, generating either images or videos based on the media type."""
    def __init__(
        self, 
        media_type: str,
        mri_data: MRISlices, 
        config: PlotConfig, 
        output_dir: str,
        scan_name: str, 
        underlay_image: Optional[MRISlices] = None
    ):
        """
        Args:
            media_type (str): File format for the output (e.g., 'png', 'mp4').
            mri_data (MRISlices): MRI data to be plotted.
            config (PlotConfig): Configuration settings for plotting.
            output_dir (str): Directory to save the output.
            scan_name (str): Base name for the output files.
            underlay_image (Optional[MRISlices]): Underlay image for overlays.
        """
        self.media_type = media_type
        self.mri_data = mri_data
        self.config = config
        self.output_dir = output_dir
        self.scan_name = scan_name
        self.underlay_image = underlay_image

        os.makedirs(self.output_dir, exist_ok=True)

    def plot(self):
        """Plots the MRI data based on the specified media type."""
        if self.media_type == "png":
            self._plot_images()
        elif self.media_type == "mp4":
            self._plot_videos()
        else:
            raise ValueError(f"Unsupported media type: {self.media_type}. Only 'png' and 'mp4' are supported.")

    def _plot_images(self):
        """Generates and saves images for three orthogonal planes."""
        plane_images = self.mri_data.add_titles_and_generate_images(
            config=self.config, title_prefix=self.scan_name, single_timepoint=True
        )

        for plane, image in plane_images.items():
            output_path = os.path.join(self.output_dir, f"{self.scan_name}_{plane}.png")
            imageio.imwrite(output_path, image)

    def _plot_videos(self):
        """Generates videos for three orthogonal planes."""
        video_writers = self._initialize_video_writers()

        # Iterate over timepoints
        for t in range(self.mri_data.shape[-1]): # Assumes last dimension is timepoint
            plane_images = self.mri_data.add_titles_and_generate_images(
                config=self.config,
                title_prefix=self.scan_name,
                single_timepoint=False,
                slice_timepoint=t,
                underlay_slice=self.underlay_image,
            )
            
            # Write all frames by appending images to video writers
            for plane, image in plane_images.items():
                video_writers[plane].append_data(image)

        # Close all video writers
        for writer in video_writers.values():
            writer.close()

    def _initialize_video_writers(self):
        """Initializes video writers for each plane."""
        return {
            plane: imageio.get_writer(
                os.path.join(self.output_dir, f"{self.scan_name}_{plane}.mp4"),
                fps=self.config.fps,
                codec="libx264",
            )
            for plane in ["sagittal", "coronal", "axial"]
        }

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
                            media_type=processor.media_type,
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