import os
import sys
import argparse
import logging
from typing import List, Optional
from extractUtils import (
    extract_group_displacements,
    convert_displacements_to_dataframe,
    get_subcortical_rois,
    get_ventricle_rois,
    get_valid_rois,
    save_displacement_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extractMotion.log'),
        logging.StreamHandler()
    ]
)

def extract_and_save_displacements(bids_dir: str, mni_tf: bool, rois: Optional[List[str]], output_path: str,
                                 collect_all: bool = False, collect_subcortical: bool = False,
                                 collect_ventricles: bool = False, noise: bool = False, save_as_r: bool = False) -> None:
    """
    Extracts displacements from a specified BIDS directory, converts the results to a DataFrame,
    and saves it to a CSV or RDS file.

    Parameters:
    - bids_dir: str
        Path to the BIDS directory containing subject data.
    - mni_tf: bool
        Boolean indicating whether to use MNI-space displacements or parcellation-based displacements.
    - rois: Optional[List[str]]
        List of region of interest names for parcellation-based displacements.
    - output_path: str
        Path where the resulting DataFrame will be saved.
    - collect_all: bool
        If True, collect displacements for all available ROIs.
    - collect_subcortical: bool
        If True, collect displacements for all subcortical regions.
    - collect_ventricles: bool
        If True, collect displacements for all ventricle regions.
    - noise: bool
        If True, trigger a custom extraction process for noise.
    - save_as_r: bool
        If True, save as RDS format, otherwise save as CSV.
    """
    # Ensure the BIDS directory exists
    if not os.path.isdir(bids_dir):
        logging.error(f"The specified BIDS directory '{bids_dir}' does not exist.")
        sys.exit(1)
        
    if not output_path.endswith('.rds') and not output_path.endswith('.csv'):
        raise ValueError("Invalid file extension. Please use .rds or .csv.")

    # Handle ROI collection based on flags
    if not mni_tf:
        if collect_all:
            valid_rois, _ = get_valid_rois()
            rois = valid_rois
        elif collect_subcortical and collect_ventricles:
            rois = get_subcortical_rois() + get_ventricle_rois()
        elif collect_subcortical:
            rois = get_subcortical_rois()
        elif collect_ventricles:
            rois = get_ventricle_rois()
        
        if not rois:
            logging.error("When 'mni_tf' is False, either 'rois' must be specified or one of the collection flags must be set.")
            sys.exit(1)

    # Run the extraction
    displacements = extract_group_displacements(bids_dir, mni_tf, rois if not mni_tf else [], noise)
    logging.info("Displacements successfully extracted.")
    
    # Convert the displacements dictionary to a DataFrame
    displacement_df = convert_displacements_to_dataframe(displacements)

    # Save the DataFrame using the utility function
    save_displacement_data(displacement_df, output_path)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="""Extract and save displacement data from a BIDS directory.
    
    This script extracts motion displacement data from MRI scans in a BIDS directory structure.
    It can work with either MNI-space displacements or parcellation-based displacements.
    
    For parcellation-based displacements (when --mni_tf is not set), you can either:
    1. Specify a list of ROIs using --rois
    2. Use --collect_all to collect data from all available ROIs
    3. Use --collect_subcortical to collect data from all subcortical regions
    4. Use --collect_ventricles to collect data from all ventricle regions
    
    The subcortical regions include:
    - Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala, Accumbens-area, and VentralDC
    (both left and right hemispheres)
    
    The ventricle regions include:
    - Lateral Ventricle, Inferior Lateral Ventricle, 3rd Ventricle, 4th Ventricle, 5th Ventricle
    (both left and right hemispheres where applicable)
    """)
    
    parser.add_argument("bids_dir", type=str, help="Path to the BIDS directory containing subject data.")
    parser.add_argument("output", type=str, help="Path to save the output file (CSV or RDS).")
    parser.add_argument("--mni_tf", action="store_true", help="Use MNI-space displacements (set for True), or parcellation-based displacements if False.")
    parser.add_argument("--rois", nargs="*", help="List of region of interest names for parcellation-based displacements (required if mni_tf is False and no collection flags are set).")
    parser.add_argument("--collect_all", action="store_true", help="Collect displacement data from all available ROIs.")
    parser.add_argument("--collect_subcortical", action="store_true", help="Collect displacement data from all subcortical regions.")
    parser.add_argument("--collect_ventricles", action="store_true", help="Collect displacement data from all ventricle regions.")
    parser.add_argument("--noise", action="store_true", help="Trigger custom extraction process for noise.")

    args = parser.parse_args()

    extract_and_save_displacements(
        args.bids_dir,
        args.mni_tf,
        args.rois,
        args.output,
        args.collect_all,
        args.collect_subcortical,
        args.collect_ventricles,
        args.noise
    )
