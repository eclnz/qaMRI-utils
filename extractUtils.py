import numpy as np
import pandas as pd
import os
import warnings
from typing import List, Dict, Union, Tuple, Optional
from processingUtils import crop_to_nonzero, apply_crop_bounds
import nibabel as nib
from dcm2bids import list_bids_subjects_sessions_scans
import logging

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_displacement_data(displacement_data_5d: np.ndarray, axis_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts displacement data for a specified axis from a 5D dataset and
    calculates the mean and standard deviation across the ROI for each time point.

    Parameters:
    - displacement_data_5d: np.ndarray
        A 5D numeric array with dimensions [x, y, z, axis, time], where
        x, y, z represent spatial dimensions, axis represents different
        directional components of displacement, and time represents the time points.
    - axis_index: int
        An integer specifying the axis to extract (e.g., 1 for x, 2 for y, 3 for z).
        The axis_index should be within the range of the fourth dimension of displacement_data_5d.

    Returns:
    - Tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays:
        - mean_displacement: 1D array with mean displacements for each time point.
        - std_displacement: 1D array with standard deviations for each time point.
    """

    # Type checking
    if not isinstance(displacement_data_5d, np.ndarray) or displacement_data_5d.ndim != 5:
        raise ValueError("displacement_data_5d must be a 5D numpy array.")
    if not isinstance(axis_index, int) or not (0 <= axis_index < displacement_data_5d.shape[3]):
        raise ValueError(f"axis_index must be an integer within the range [0, {displacement_data_5d.shape[3] - 1}].")

    # Get dimensions of the input data
    _, _, _, num_axes, num_time_points = displacement_data_5d.shape

    # Check if axis_index is within the allowable range
    if axis_index >= num_axes:
        raise ValueError("Specified axis_index exceeds the dimensions of the input data.")

    # Extract data for the given axis
    displacement_data_4d = displacement_data_5d[:, :, :, axis_index, :]  # 4D data with dimensions [x, y, z, time]

    # Compute the mean and standard deviation across the ROI (non-zero voxels) for each time point
    mean_displacement = np.zeros(num_time_points)
    std_displacement = np.zeros(num_time_points)

    for t in range(num_time_points):
        # Extract the 3D displacement data for the current time point
        displacement_3d = displacement_data_4d[..., t]

        # Consider only non-zero voxels
        non_zero_voxels = displacement_3d[displacement_3d != 0]
        if non_zero_voxels.size > 0:
            mean_displacement[t] = np.mean(non_zero_voxels)
            std_displacement[t] = np.std(non_zero_voxels)
        else:
            mean_displacement[t] = 0
            std_displacement[t] = 0

    return mean_displacement, std_displacement

def extract_roi_displacement(motion_data: np.ndarray, brain_mask: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts displacement data for a specific region of interest (ROI) from motion data.
    The function crops the motion data to the bounds of the non-zero region in the brain mask,
    then calculates the mean and standard deviation of the displacement data for each axis (x, y, z) across time.

    Parameters:
    - motion_data: np.ndarray
        A 5D array containing displacement data with dimensions [x, y, z, axis, time].
    - brain_mask: np.ndarray
        A 3D binary matrix representing the brain mask, where non-zero elements indicate the region of interest.

    Returns:
    - Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        A tuple containing three tuples, one for each axis (x, y, z).
        Each tuple contains:
        - mean_displacement: 1D array with mean displacements for each time point.
        - std_displacement: 1D array with standard deviations for each time point.
    """
    
    # Type checking
    if not (isinstance(motion_data, np.ndarray) and motion_data.ndim == 5):
        raise ValueError("motion_data must be a 5D numpy array.")
    if not (isinstance(brain_mask, np.ndarray) and brain_mask.ndim == 3 and (brain_mask.dtype == bool or np.issubdtype(brain_mask.dtype, np.integer))):
        raise ValueError("brain_mask must be a 3D binary (boolean or integer) numpy array.")
    
    # Ensure that the first three dimensions of motion_data and brain_mask match
    if motion_data.shape[:3] != brain_mask.shape:
        raise ValueError(f"Size mismatch: motion_data has size {motion_data.shape[:3]}, while brain_mask has size {brain_mask.shape}.")

    # Calculate cropping bounds from the mask
    crop_bounds = crop_to_nonzero(brain_mask)
    
    # Set voxels outside of the mask to zero
    motion_in_mask = motion_data * brain_mask[..., np.newaxis, np.newaxis]
    
    # Crop the motion data according to the calculated bounds
    brain_motion_cropped = apply_crop_bounds(motion_in_mask, crop_bounds)

    # Extract mean and standard deviation displacement data for each axis
    x_mean, x_std = extract_displacement_data(brain_motion_cropped, axis_index=0)
    y_mean, y_std = extract_displacement_data(brain_motion_cropped, axis_index=1)
    z_mean, z_std = extract_displacement_data(brain_motion_cropped, axis_index=2)

    return (x_mean, x_std), (y_mean, y_std), (z_mean, z_std)

def extract_subject_parcellation_displacements(motion_file: str, parcellation_file: str, roi_names: List[str]) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Extracts motion displacements for a subject from a given motion file using
    specific ROIs within a parcellation image. Calculates the mean and standard deviation
    of displacement for each ROI and each time point based on the specified labels
    in the parcellation image.

    Parameters:
    - motion_file: str
        Path to the motion file in NIfTI format from which displacements are to be extracted.
    - parcellation_file: str
        Path to the parcellation image in NIfTI format from which ROIs are to be extracted.
    - roi_names: List[str]
        List of names for the regions of interest in the parcellation image.

    Returns:
    - Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        A dictionary where each key is a region label, and each value is a dictionary with:
            - 'x': (mean_displacement, std_displacement) for the x-axis
            - 'y': (mean_displacement, std_displacement) for the y-axis
            - 'z': (mean_displacement, std_displacement) for the z-axis
    """
    
    logging.info("Starting extraction of subject parcellation displacements.")
    
    # Validate that files are in NIfTI format
    if not motion_file.endswith(('.nii', '.nii.gz')) or not parcellation_file.endswith(('.nii', '.nii.gz')):
        logging.error("Both motion_file and parcellation_file must be NIfTI files with extensions .nii or .nii.gz.")
        raise ValueError("Both motion_file and parcellation_file must be NIfTI files with extensions .nii or .nii.gz.")

    # Load valid ROIs and their corresponding labels
    valid_rois, roi_labels = get_valid_rois()

    # Find matched ROIs and their corresponding labels
    matched_rois = {roi: label for roi, label in zip(valid_rois, roi_labels) if roi in roi_names}
    unmatched_rois = set(roi_names) - set(matched_rois.keys())

    if not roi_labels:
        raise ValueError("When 'mni_tf' is False, either 'rois' must be specified or one of the collection flags must be set.")

    if unmatched_rois:
        raise ValueError(f"The following ROI names do not match any valid ROIs and will be ignored: {', '.join(unmatched_rois)}")
    
    if not matched_rois:
        raise ValueError("None of the provided ROI names match the valid ROIs.")

    # Read the motion and parcellation images
    motion_img = nib.load(motion_file)
    parcellation_img = nib.load(parcellation_file)

    motion_data = motion_img.get_fdata()  # type: ignore
    parcellation_data = parcellation_img.get_fdata()  # type: ignore

    if motion_data.shape[:3] != parcellation_data.shape:
        raise ValueError("The first three spatial dimensions of the motion and parcellation files must match.")

    # Initialize dictionary to store displacements
    displacements = {}

    # Process each matched ROI
    for region_name, region_label in matched_rois.items():
        logging.info(f"Processing ROI: {region_name} (Label: {region_label})")  # Log each ROI being processed
        # Create a binary mask for the current region
        region_mask = parcellation_data == region_label

        # Mask the motion data to isolate the region of interest
        roi_motion_data = motion_data * region_mask[..., np.newaxis, np.newaxis]

        try:
            # Compute mean and standard deviation of displacement along each axis (x, y, z)
            x_mean, x_std = extract_displacement_data(roi_motion_data, axis_index=0)
            y_mean, y_std = extract_displacement_data(roi_motion_data, axis_index=1)
            z_mean, z_std = extract_displacement_data(roi_motion_data, axis_index=2)

            displacements[region_name] = {
                'x': (x_mean, x_std),
                'y': (y_mean, y_std),
                'z': (z_mean, z_std)
            }
        except Exception as e:
            logging.error(f"Error processing ROI: {region_name} (Label: {region_label})")
            logging.error(f"Error: {e}")

    logging.info("Extraction of subject parcellation displacements completed successfully.")
    
    return displacements

def get_subcortical_rois() -> List[str]:
    """
    Returns a list of subcortical ROI names.

    Returns:
    - List[str]: List of subcortical region names
    """
    return [
        'Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
        'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Left-VentralDC',
        'Right-Thalamus', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum',
        'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC'
    ]

def get_ventricle_rois() -> List[str]:
    """
    Returns a list of ventricle ROI names.

    Returns:
    - List[str]: List of ventricle region names
    """
    return [
        'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', '3rd-Ventricle', '4th-Ventricle',
        'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', '5th-Ventricle'
    ]

def get_valid_rois() -> Tuple[List[str], List[int]]:
    """
    Returns valid ROI names and their corresponding labels.

    Returns:
    - Tuple[List[str], List[int]]
        A tuple of ROI names and their corresponding labels.
    """
    roi_names = [
        'Unknown', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex', 'Left-Lateral-Ventricle',
        'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex', 'Left-Thalamus',
        'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem',
        'Left-Hippocampus', 'Left-Amygdala', 'CSF', 'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
        'Left-choroid-plexus', 'Right-Cerebral-White-Matter', 'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle',
        'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus',
        'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area',
        'Right-VentralDC', 'Right-vessel', 'Right-choroid-plexus', '5th-Ventricle', 'WM-hypointensities',
        'Left-WM-hypointensities', 'Right-WM-hypointensities', 'non-WM-hypointensities', 'Left-non-WM-hypointensities',
        'Right-non-WM-hypointensities', 'Optic-Chiasm', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central',
        'CC_Mid_Anterior', 'CC_Anterior'
    ]

    roi_values = [
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50,
        51, 52, 53, 54, 58, 60, 62, 63, 72, 77, 78, 79, 80, 81, 82, 85, 251, 252, 253, 254, 255
    ]

    return roi_names, roi_values

def extract_subject_mni_displacements(motion_file: str) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Extracts motion displacements for a subject from a given motion file.
    Calculates the mean and standard deviation of displacement for each
    region of interest (ROI) in predefined region masks.

    Parameters:
    - motion_file: str
        Path to the motion file in NIfTI format from which displacements are to be extracted.

    Returns:
    - Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        A dictionary where each key is a region name and the value is a dictionary containing
        'x', 'y', and 'z' displacements (each with mean and standard deviation) for that region.
    """
    
    region_mask_dir = '../qaMRI-clone/template/region_masks'

    # Check if the region mask directory exists
    if not os.path.isdir(region_mask_dir):
        raise ValueError(f"The specified region mask directory '{region_mask_dir}' does not exist.")

    # Load the motion image
    motion_img = nib.load(motion_file)
    motion_data = motion_img.get_fdata()  # type: ignore

    # Initialize dictionary to store displacements
    displacements = {}

    # Get a list of all .nii.gz files in the region_mask_dir
    mask_files = [f for f in os.listdir(region_mask_dir) if f.endswith('.nii.gz')]

    # Loop through each region mask and calculate displacements
    for mask_file in mask_files:
        # Extract the region name from the file name (without extension)
        region_name = os.path.splitext(os.path.splitext(mask_file)[0])[0]
        
        # Read the ROI mask
        mask_path = os.path.join(region_mask_dir, mask_file)
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()  # type: ignore

        # Ensure the dimensions match
        if motion_data.shape[:3] != mask_data.shape:
            raise ValueError(f"Dimension mismatch between motion file and mask file '{mask_file}' for region '{region_name}'.")

        # Apply the mask to the motion data for the specific region
        roi_motion_data = motion_data * mask_data[..., np.newaxis, np.newaxis]

        # Calculate the mean and standard deviation for each axis
        x_mean, x_std = extract_displacement_data(roi_motion_data, axis_index=0)
        y_mean, y_std = extract_displacement_data(roi_motion_data, axis_index=1)
        z_mean, z_std = extract_displacement_data(roi_motion_data, axis_index=2)

        # Store the results in the displacements dictionary
        displacements[region_name] = {
            'x': (x_mean, x_std),
            'y': (y_mean, y_std),
            'z': (z_mean, z_std)
        }

    return displacements

def extract_group_displacements(bids_dir: str, mni_tf: bool, rois: Optional[List[str]], noise: bool) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]]]:
    """
    Extracts motion displacements for a group of subjects from motion files within a BIDS directory structure.

    Parameters:
    - bids_dir: str
        Path to the BIDS directory containing subject data.
    - mni_tf: bool
        Boolean indicating whether to use MNI-space displacements or parcellation-based displacements.
    - rois: Optional[List[str]]
        List of region of interest names for parcellation-based displacements. Required if mni_tf is False.
    - noise: bool
        Boolean indicating whether to use noise-based displacements.

    Returns:
    - Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]]]
        A nested dictionary with mean and standard deviation displacements for each subject, session, and ROI.
    """

    # Determine the appropriate file location and extension 
    if noise: # Extract noise-based displacements
        file_extension = 'motion_map'
        location = os.path.join(bids_dir,'derivatives','noise')
        motion_images = list_bids_subjects_sessions_scans(location, file_extension)
    elif mni_tf: # Extract MNI-space displacements
        file_extension = 'motion.nii.gz'
        location = os.path.join(bids_dir,'derivatives','registration')
        motion_images = list_bids_subjects_sessions_scans(location, file_extension)
    else: # Extract parcellation-based displacements
        file_extension = 'motion_map.nii.gz'
        location = os.path.join(bids_dir,'derivatives','aMRI')
        motion_images = list_bids_subjects_sessions_scans(location, file_extension)

    # Initialize dictionary to store subject motion displacements
    subject_motion_displacements = {}

    # Iterate over each subject in the motion images dictionary
    for subject, sessions in motion_images.items():
        subject_displacements = {}

        # Iterate over each session for the subject
        for session, scans in sessions.items():
            logging.info(f"Processing {subject} {session}")
            session_displacements = {}
            
            # Iterate over each scan in the session
            for scan_name, scan_info in scans.items():
                logging.info(f"Processing scan: {scan_name}")
                
                if noise:
                    try:
                        motion_file = scan_info["scan_path"]
                        file_name = os.path.basename(motion_file)
                        noise_level = file_name.split('_')[4].split('-')[1]
                        repeat = file_name.split('_')[5].split('-')[1].split('.')[0]
                    except (IndexError, KeyError) as e:
                        logging.error(f"Error parsing noise scan information: {e}")
                        continue
                else:
                    motion_file = scan_info["scan_path"]

                try:
                    # Extract displacements using the appropriate function
                    if mni_tf:
                        displacements = extract_subject_mni_displacements(motion_file)
                    else:
                        # Generate the path to the parcellation file
                        parcellation_file = os.path.join(bids_dir, 'derivatives', 'segmentation', 
                                                       subject, session, 
                                                       f"{subject}_{session}_desc-padded_segmentation.nii.gz")
                        
                        # Check if the parcellation file exists
                        if not os.path.isfile(parcellation_file):
                            logging.warning(f"Segmentation image missing: {parcellation_file}")
                            continue

                        # Extract displacements for the specified ROIs using the parcellation method
                        if rois is None:
                            raise ValueError("ROIs must be specified when using parcellation-based displacements")
                        displacements = extract_subject_parcellation_displacements(motion_file, parcellation_file, rois)

                    # Store displacements for each scan with metadata
                    scan_data = {
                        'displacements': displacements,
                        'metadata': {
                            'noise_level': noise_level,
                            'repeat': repeat
                        } if noise else {}
                    }
                    session_displacements[scan_name] = scan_data
                    
                except Exception as e:
                    logging.error(f"Error processing scan {scan_name}: {e}")
                    continue

            # Only store session data if we have processed at least one scan successfully
            if session_displacements:
                subject_displacements[session] = session_displacements

        # Only store subject data if we have processed at least one session successfully
        if subject_displacements:
            subject_motion_displacements[subject] = subject_displacements

    return subject_motion_displacements #type: ignore

def convert_displacements_to_dataframe(subject_motion_displacements: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]]]) -> pd.DataFrame:
    """
    Converts a nested dictionary structure of motion displacements into a pandas DataFrame
    for easier analysis and export.

    Parameters:
    - subject_motion_displacements: Dict
        Nested dictionary with mean and standard deviation displacements for each subject, session, and ROI.

    Returns:
    - pd.DataFrame
        A DataFrame where each row corresponds to a subject, session, ROI, axis, and time point,
        containing mean and standard deviation displacement values and metadata.
    """
    # Initialize a list to hold rows for the DataFrame
    rows = []

    # Iterate over the nested dictionary structure
    for subject, sessions in subject_motion_displacements.items():
        for session, session_data in sessions.items():
            for scan_name, scan_data in session_data.items():
                for roi, roi_data in scan_data['displacements'].items():
                    for axis, axis_data in roi_data.items(): #type: ignore
                        mean_displacement, std_displacement = axis_data  # Unpack the tuple
                        # Create a row for each time point
                        for timepoint, (mean_val, std_val) in enumerate(zip(mean_displacement, std_displacement)):
                            row_data = {
                                'subject': subject,
                                'session': session,
                                'scan': scan_name,
                                'roi': roi,
                                'axis': axis,
                                'timepoint': timepoint,
                                'mean_displacement': mean_val,
                                'std_displacement': std_val
                            }
                            # Add metadata if it exists
                            if 'metadata' in scan_data:
                                row_data.update(scan_data['metadata'])
                            
                            rows.append(row_data)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)
    return df