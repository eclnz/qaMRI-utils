import os
import json
import csv
import tempfile
import zipfile
import pydicom
from pydicom import DataElement
import subprocess
import shutil
import warnings
from collections import defaultdict
import curses
import textwrap
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, DefaultDict, BinaryIO, Union, Any
from datetime import datetime
from helperFunctions import sanitize_string
import io
from collections import defaultdict
import logging
import argparse

# Configure logging
def setup_logging(verbose: bool = False):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Define categories and their keywords
SCAN_CATEGORIES = {
    "anat": ["t1", "t2", "flair", "amri", "anatomy", "bravo", "inhance", "stir", "sag"],
    "dwi": ["dwi", "dki", "dsir", "dsires", "diffusion"],
    "func": ["rs_", "resting", "fmri", "default_mode", "attention_network", "visual_network"],
    "maps": ["map", "b0", "t1_map", "t2_map", "pd_map", "quantification"],
    "asl": ["easl", "cbf", "transit_corrected", "perfusion", "asl"],
    # Catch any scans that might not fit a primary modality but are still valid in BIDS
    # "derivatives": ["processed", "synthetic", "segmentation", "screenshot", "sy", "mean_epi"],
    "fmaps": ["phasediff", "magnitude", "fieldmap", "b0"],
    # Add other specific modalities as needed (e.g., spectroscopy, PET, etc.)
    "pet": ["pet", "suvr", "amyloid"],
}

def allocate_scan_to_group(scan_name: str) -> str:
    """Allocates a scan to its appropriate group based on keywords."""
    scan_name_lower = scan_name.lower()

    for category, keywords in SCAN_CATEGORIES.items():
        if any(keyword in scan_name_lower for keyword in keywords):
            return category

    # Default category for unmatched scans
    return "other"

def parse_study_description(study_description: str, patient_name: str):
    """
    Parses the StudyDescription to extract cohort, subject ID, and session ID.
    Falls back to parsing patientName if StudyDescription appears incorrectly formatted.
    
    Args:
        study_description (str): The StudyDescription header from DICOM metadata.
        patient_name (str): The PatientName field, used as a fallback.
    
    Returns:
        tuple: (cohort, subject_id, session_id)
    """
    def parse_parts(parts):
        """Helper function to extract cohort, subject_id, and session from parts."""
        cohort = None
        subject_id = None
        session = None
        
        for part in reversed(parts):
            if part.isdigit() and subject_id is None:  # Subject ID is numeric
                subject_id = int(part)
            elif part.isupper() and len(part) == 1 and session is None:  # Session is a single uppercase letter
                session = part
            elif part.isalpha() and cohort is None:  # Cohort is alphabetic
                cohort = part.lower()
        
        return cohort, subject_id, session

    # Split the StudyDescription into parts
    parts = study_description.split("_")
    
    # If parts length <= 3, fallback to parsing patientName
    if len(parts) <= 3:
        # print(f"Warning: StudyDescription '{study_description}' seems invalid, falling back to PatientName.")
        parts = patient_name.split("_")
    
    # Parse parts for cohort, subject ID, and session
    cohort, subject_id, session = parse_parts(parts)
    
    # Fill with default values if not found
    cohort = cohort or "exp"  # Default cohort
    session = session or "A"  # Default session
    
    # Validate subject ID is found
    if subject_id is None:
        subject_id = hash(patient_name) % 1000
        # raise ValueError(f"Subject ID not found in StudyDescription or PatientName: {study_description}, {patient_name}")
    
    return cohort, subject_id, session

def dcm2bids(data_directory: str, bids_output_dir: str, zip: bool = False, force_slice_thickness: bool = False, conversion_method: str = 'dcm2niix'):
    bids_raw_output_dir = os.path.join(bids_output_dir, 'raw')
    
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    if not os.path.exists(bids_output_dir):
        os.mkdir(bids_output_dir)
    
    # List all subjects, sessions, and scans containing the specified file type
    subjects_sessions_scans = list_non_bids_subjects_sessions_scans(data_directory, zip)

    # Build the series structure and get unique series descriptions
    unique_series = build_series_list(subjects_sessions_scans)
    
    if not unique_series:
        if not zip:
            # If no scans were found and not searching within zips, retry search with zip enabled
            subjects_sessions_scans = list_non_bids_subjects_sessions_scans(data_directory, True)
            unique_series = build_series_list(subjects_sessions_scans)
        
        # Check again after retrying with zip
        if not unique_series:
            display_error("No scans were found in the specified directory")
            return None

    # Display dropdown menu to select series descriptions
    selected_series = display_dropdown_menu(unique_series, title_text="Select scans to add to BIDS")
    
    if len(selected_series) == 0:
        display_error("No scans selected")
        return None

    logging.info("Selected series descriptions for processing:")
    for series in selected_series:
        logging.info(f" - {series}")
        
    # Process only scans that match the selected series descriptions
    for subject_id, sessions in subjects_sessions_scans.items():
        # Include cohort in subject name 
        cohort = "unknown"
        if sessions:  # Check if sessions dictionary is not empty
            first_session = next(iter(sessions.values()))
            if first_session:  # Check if first session has scans
                first_scan = next(iter(first_session.values()))
                if first_scan and "cohort" in first_scan:
                    cohort = first_scan["cohort"]
                    
        subject_name = f"sub-{cohort}{subject_id}"
        logging.info(f"Processing Subject: {subject_name}")

        for session_id, scans in sessions.items():
            # Extract the year from the session metadata
            year = scans[next(iter(scans))]["date"] if scans else "unknown"
            session_name = f"ses-{year}{session_id}"
            logging.info(f"  Processing Session: {session_name}")
            
            if len(scans.items()) ==0: 
                display_error(f"No scans: {selected_series} found for {subject_name} {session_name}")
                continue

            for scan, metadata in scans.items():
                series_description = scan  # The scan key represents the series description

                # Skip scans that are not in the selected series
                if series_description not in selected_series:
                    continue

                raw_path = metadata["dicom_path"]
                
                use_temp_dir = False

                try:
                    # Standardize the scan name based on series description
                    scan_group = allocate_scan_to_group(series_description)
                    standardized_scan_name = standardize_scan_name(series_description)

                    out_path = os.path.join(bids_raw_output_dir, subject_name, session_name, scan_group)

                    # Check if the raw path is from a ZIP
                    if '.zip:' in raw_path:
                        # Split the ZIP path and internal file path
                        zip_path, internal_path = raw_path.split('.zip:')
                        zip_path += '.zip'  # Add back the ".zip" extension
                        
                        use_temp_dir = True

                        # Create a temporary directory for extraction
                        temp_dir = tempfile.mkdtemp()
                        raw_folder = extract_dicom_from_zip(zip_path, internal_path, temp_dir)
                        
                        for file in Path(raw_folder).iterdir():
                            if file.is_file():  # Check if it's a file (not a directory)
                                if is_valid_dicom(str(file)):  # Check if the file is a valid DICOM
                                    raw_path = str(file)  # Set raw_path to the first valid DICOM
                                    break  # Stop looping once a valid DICOM is found

                        if not raw_folder:
                            raise ValueError(f"Failed to extract files from ZIP: {zip_path}")

                    
                    process_scan(
                        raw_path, out_path, subject_name, session_name,
                        standardized_scan_name, force_slice_thickness=force_slice_thickness, conversion_method=conversion_method
                    )
                    
                except Exception as e:
                    logging.error(f"    Failed to process scan {scan}. Error: {e}")

                finally:
                    # Clean up the temporary directory if it was created
                    if use_temp_dir:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)

    setup_sheets(bids_output_dir)

def ensure_binary(file_input: Union[str, bytes, BinaryIO, os.DirEntry]) -> BinaryIO:
    """
    Ensure the input is a binary file-like object.

    Args:
        file_input (Union[str, bytes, BinaryIO, os.DirEntry]): A file path, raw binary data,
        binary file-like object, or DirEntry.

    Returns:
        BinaryIO: Binary file-like object.

    Raises:
        TypeError: If the input is not a valid type.
    """
    if isinstance(file_input, os.DirEntry):  # If it's a DirEntry object
        if file_input.is_file():  # Ensure it's a file
            return open(file_input.path, "rb")
        else:
            raise TypeError(f"DirEntry {file_input.name} is not a file.")
    elif isinstance(file_input, str):  # If it's a file path
        return open(file_input, "rb")
    elif isinstance(file_input, bytes):  # If it's raw binary data
        return io.BytesIO(file_input)
    elif hasattr(file_input, "read"):  # If it's already a file-like object
        file_input.seek(0)  # Ensure the pointer is at the beginning
        return file_input
    else:
        raise TypeError("file_input must be a file path, bytes, binary file-like object, or DirEntry.")

def extract_dicom_from_zip(zip_path: str, internal_path: str, temp_dir: str) -> str:
    """
    Extract a DICOM file or folder from a ZIP file into a temporary directory.

    Args:
        zip_path (str): Path to the ZIP file.
        internal_path (str): Path to the file or folder inside the ZIP.
        temp_dir (str): Path to the temporary directory for extraction.

    Returns:
        str: Path to the extracted folder in the temporary directory.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Find all files in the same folder as the internal path
            folder = '/'.join(internal_path.split('/')[:-1])
            relevant_files = [entry for entry in zip_file.namelist() if entry.startswith(folder)]

            # Extract the relevant files into the temporary directory
            for file in relevant_files:
                zip_file.extract(file, temp_dir)

            # Return the extracted folder path
            extracted_folder = os.path.join(temp_dir, folder)
            return os.path.abspath(extracted_folder)

    except Exception as e:
        print(f"Error extracting DICOM files from {zip_path}: {e}")
        return ""

def build_series_list(subjects_sessions_scans: Dict[str, Dict[str, Dict[str, Dict[str, str]]]]) -> list:
    """
    Finds all unique scan names in the subjects_sessions_scans structure.

    Args:
        subjects_sessions_scans (Dict): The nested structure of subjects, sessions, and scans.

    Returns:
        set: A set of unique scan names.
    """
    unique_scans = set()

    for subject_id, sessions in subjects_sessions_scans.items():
        for session_id, scans in sessions.items():
            for scan in scans:
                if scan != "cohort":  # Ignore the cohort key
                    unique_scans.add(scan)

    return sorted(unique_scans)

def list_bids_subjects_sessions_scans(data_directory: str, file_extension: str, raw: bool = False) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Recursively traverses directories to list files by subject, session, and scan in a BIDS-compliant structure.

    Args:
        data_directory (str): Path to the base directory containing files.
        file_extension (str): File extension to look for (e.g., '.nii.gz').

    Returns:
        Dict[str, Dict[str, Dict[str, Dict[str, str]]]]: A dictionary with subjects, sessions, and scans containing metadata.
    """
    subjects_sessions_scans: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    if not Path(data_directory).is_dir():
        raise ValueError(f"Data directory '{data_directory}' does not exist or is not a directory.")

    def recursive_traverse(path: Path):
        """
        Recursively traverses the directory structure to detect subjects, sessions, and scans.

        Args:
            path (Path): Current directory path to process.
        """
        logging.debug(f"Traversing directory: {path}")
        for entry in path.iterdir():
            if entry.is_dir():
                logging.debug(f"Found directory: {entry}")
                # Handle subject directories
                if entry.name.startswith("sub-"):
                    recursive_traverse(entry)  # Process sessions within the subject

                # Handle session directories
                elif entry.name.startswith("ses-"):
                    subject_dir = entry.parent.name
                    if subject_dir.startswith("sub-"):
                        recursive_traverse(entry)  # Process scans within the session

                # Traverse deeper for other directories
                else:
                    recursive_traverse(entry)

            elif entry.is_file() and (entry.name.endswith(file_extension) or file_extension in entry.name):
                # Extract metadata
                parent_session = entry.parent.name
                parent_subject = entry.parent.parent.name

                if raw:
                    # Extract metadata with flexibility for folder structure
                    parent_session = entry.parent.parent.name if entry.parent.parent else "unknown"
                    parent_subject = entry.parent.parent.parent.name if entry.parent.parent and entry.parent.parent.parent else "unknown"

                # Ensure the hierarchy is valid
                if not parent_subject.startswith("sub-"):
                    parent_subject = "unknown"  # Fallback for subject

                if not parent_session.startswith("ses-"):
                    parent_session = "unknown"  # Fallback for session

                # Skip files that cannot be matched to a valid subject/session structure
                if parent_subject == "unknown" or parent_session == "unknown":
                    continue

                # Extract scan description
                parts = entry.name.split("_desc-")
                if len(parts) > 1:
                    scan = parts[1]
                else:
                    scan = entry.name

                # Populate the structure
                subjects_sessions_scans[parent_subject][parent_session][scan]["scan_path"] = os.path.join(path, entry.name)

    # Start recursive traversal
    recursive_traverse(Path(data_directory))

    # Convert defaultdict to standard dictionary for cleaner return
    return {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in subjects_sessions_scans.items()}

def is_valid_dicom(file_obj):
        """
        Check if a file is a valid DICOM by inspecting its header.
        """
        file_obj = ensure_binary(file_obj)
        try:
            file_obj.seek(0)  # Ensure the file-like object starts from the beginning
            dcm = pydicom.dcmread(file_obj, stop_before_pixels=True, force=True)
            # Check for required DICOM attributes
            return hasattr(dcm, 'PatientID') and hasattr(dcm, 'StudyInstanceUID')
        except Exception:
            return False

def list_non_bids_subjects_sessions_scans( # TODO: Could be better as a class.
    data_directory: str, zip: bool
) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Recursively traverses directories to list files by subject, session, and scan.
    
    Args:
        data_directory (str): Path to the base directory containing files.
        file_extension (str): File extension to look for (default: '.dcm').
    
    Returns:
        Dict[str, Dict[str, Dict[str, List[str]]]]: A dictionary with subjects, sessions,
        and scans containing file paths.
    """

    subjects_sessions_scans: DefaultDict[str, DefaultDict[str, DefaultDict[str, DefaultDict[str, str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str))))

    def process_dicom(file: Union[str, io.BytesIO], path_descriptor=""):
        """
        Process a DICOM file and organize it into the data structure.
        """
        file_obj = ensure_binary(file)
        try:
            file_obj.seek(0)  # Ensure the file-like object starts from the beginning
            dicom = pydicom.dcmread(file_obj, force=True)

            # Extract required fields
            dicom_info = {
                "StudyDescription": getattr(dicom, "StudyDescription", ""),
                "PatientName": getattr(dicom, "PatientName", ""),
                "SeriesDescription": getattr(dicom, "SeriesDescription", "" ),
                "AcquisitionDate": getattr(dicom, "AcquisitionDate", ""),
            }

            # Split Description into components
            cohort, subject_id, session_id = parse_study_description(dicom_info["StudyDescription"], dicom_info['PatientName'].family_name) #type: ignore
            
            scan = sanitize_string(dicom_info["SeriesDescription"])

            # Convert the string to a date object
            try:
                acquisition_date = dicom_info.get("AcquisitionDate", "")
                year = str(datetime.strptime(acquisition_date, "%Y%m%d").year) if acquisition_date else ""
            except (ValueError, TypeError):
                year = ""

            # Organize the file into the structure
                # Organize the file into the structure
            subjects_sessions_scans[subject_id][session_id][scan]["dicom_path"] = path_descriptor
            subjects_sessions_scans[subject_id][session_id][scan]["cohort"] = cohort
            subjects_sessions_scans[subject_id][session_id][scan]["date"] = year

            return True
        except Exception as e:
            print(f"Error processing DICOM file {path_descriptor}: {e}")
            return False

    def process_zip(zip_path: str) -> bool:
        """
        Process the contents of a ZIP file by checking for a valid DICOM in each folder.

        Args:
            zip_path (str): Path to the ZIP file.

        Returns:
            bool: True if at least one DICOM file is found and processed, False otherwise.
        """
        logging.debug(f"Processing ZIP file: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                entries = zip_file.namelist()
                processed_folders = set()
                found_dicom = False

                for file_name in entries:
                    if file_name.endswith('/'):
                        continue

                    folder = '/'.join(file_name.split('/')[:-1])

                    if folder in processed_folders:
                        continue

                    with zip_file.open(file_name) as file:
                        if is_valid_dicom(file):
                            logging.debug(f"Valid DICOM found in ZIP: {file_name}")
                            if process_dicom(file, path_descriptor=f"{zip_path}:{file_name}"): #type: ignore
                                processed_folders.add(folder)
                                found_dicom = True

            return found_dicom

        except Exception as e:
            logging.error(f"Error processing ZIP file {zip_path}: {e}")
            return False

    def process_file(entry, zip):
        """
        Process a single file entry, either as a DICOM or a ZIP file.
        """
        logging.debug(f"Processing file: {entry.path}")
        if entry.name.endswith('.zip') and zip:
            logging.debug(f"Found ZIP file: {entry.path}")
            return process_zip(entry.path)
        if is_valid_dicom(entry.path):
            logging.debug(f"Valid DICOM file: {entry.path}")
            return process_dicom(entry.path, path_descriptor=entry.path)
        logging.debug(f"File is not a valid DICOM: {entry.path}")
        return False

    def recursive_search(folder_path: str, zip: bool) -> bool:
        """
        Recursively searches for DICOM files in the directory and organizes them.
        """
        logging.debug(f"Searching directory: {folder_path}")
        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.name.startswith('.'):
                        logging.debug(f"Skipping hidden file or directory: {entry.name}")
                        continue

                    if entry.is_dir():
                        logging.debug(f"Found directory: {entry.path}")
                        if recursive_search(entry.path, zip):
                            continue

                    elif entry.is_file():
                        logging.debug(f"Found file: {entry.path}")
                        if process_file(entry, zip):
                            return True

        except Exception as e:
            logging.error(f"Error accessing folder {folder_path}: {e}")
        
        return False

    # Start recursive search from the root directory
    recursive_search(data_directory, zip)

    # Convert defaultdict to standard dictionary for return
    return subjects_sessions_scans # TODO make this a dict not a default dict.

def display_dropdown_menu(str_list: List[str], title_text: str):
    """
    Display a dropdown menu to select multiple series descriptions using arrow keys, Space to select, and Enter to confirm.
    """
    if not str_list:
        display_error("No options were provided")
        return []  # Early exit if the input list is empty

    selected_index = 0
    selected_items:set = set()

    def navigate_menu(stdscr):
        nonlocal selected_index
        curses.curs_set(0)  # Hide the cursor

        # Title text that may wrap based on the window width
        instructions = " (use arrows to navigate, Space to select, Enter to confirm):"
        
        while True:
            # Update terminal dimensions on each iteration to handle resizing
            height, width = stdscr.getmaxyx()
            max_display_lines = height - 2  # Leave space for title and footer

            # Wrap the title text based on the current window width
            wrapped_title = textwrap.wrap(title_text + instructions, width)
            title_lines = len(wrapped_title)

            stdscr.clear()

            # Display the wrapped title text
            for i, line in enumerate(wrapped_title):
                stdscr.addstr(i, 0, line, curses.A_BOLD)

            # Calculate the start and end index for the visible window of the series list
            start_index = max(0, selected_index - (max_display_lines - title_lines) // 2)
            end_index = min(len(str_list), start_index + max_display_lines - title_lines)

            # Display the visible window of the series list
            for i in range(start_index, end_index):
                series = str_list[i]
                prefix = "> " if i == selected_index else "  "
                selection_marker = "[x] " if i in selected_items else "[ ] "

                # Construct the line with prefix and selection marker
                line = f"{prefix}{selection_marker}{series}"

                # Truncate the line with ellipsis if it exceeds the current width
                if len(line) >= width:
                    line = line[:width - 4] + "..."

                # Display the line with optional selection highlight
                try:
                    stdscr.addstr(i - start_index + title_lines, 0, line, curses.A_REVERSE if i == selected_index else 0)
                except curses.error:
                    # Skip lines that can't be displayed due to height constraints
                    pass

            stdscr.refresh()

            # Get the user's input
            key = stdscr.getch()

            # Navigation keys
            if key == curses.KEY_UP and selected_index > 0:
                selected_index -= 1
            elif key == curses.KEY_DOWN and selected_index < len(str_list) - 1:
                selected_index += 1
            elif key == ord(" "):  # Space to select/unselect
                if selected_index in selected_items:
                    selected_items.remove(selected_index)
                else:
                    selected_items.add(selected_index)
            elif key == ord("\n"):  # Enter to confirm
                if not selected_items:  # Handle case where no items are selected
                    stdscr.addstr(height - 1, 0, "No items selected. Press Enter again to confirm, or Space to select.")
                    stdscr.refresh()
                    continue  # Let the user confirm or select items
                return [str_list[i] for i in selected_items]  # Return the selected series descriptions

    selected_series = curses.wrapper(navigate_menu)
    return selected_series

def standardize_scan_name(series_description: str) -> str:
    """Standardizes scan name based on SeriesDescription in BIDS format."""
    series_description = series_description.lower()

    # Standard anatomical scans
    if 't1' in series_description and 'flair' not in series_description:
        return 'T1w'
    elif 't2' in series_description and 'flair' not in series_description:
        return 'T2w'
    elif 'flair' in series_description:
        return 'FLAIR'
    elif 'pd' in series_description:
        return 'PD'
    elif 'inplane' in series_description:
        return 'InplaneT1'
    elif 'amri' in series_description:
        return 'aMRI'

    # Diffusion-weighted imaging
    elif 'dwi' in series_description or 'diffusion' in series_description:
        return 'dwi'
    
        # Diffusion-weighted imaging
    elif 'dki' in series_description:
        return 'dki'

    # Functional scans
    elif 'bold' in series_description or 'fmri' in series_description or 'resting' in series_description:
        return 'bold'

    # Arterial Spin Labeling
    elif 'asl' in series_description or 'perfusion' in series_description:
        return 'asl'

    # Field maps
    elif 'fieldmap' in series_description or 'phasediff' in series_description:
        return 'phasediff'
    elif 'magnitude' in series_description:
        return 'magnitude'

    # Quantitative maps
    elif 'b0' in series_description:
        return 'b0map'
    elif 't1_map' in series_description:
        return 'T1map'
    elif 't2_map' in series_description:
        return 'T2map'
    elif 'pd_map' in series_description:
        return 'PDmap'

    # Spectroscopy
    elif 'mrs' in series_description or 'spectroscopy' in series_description:
        return 'mrs'

    # Catch-all for unsupported scans
    else:
        return series_description

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename by removing special characters (excluding underscores),
    replacing periods with underscores, and ensuring proper formatting.
    
    Args:
        filename (str): The original filename to sanitize.
        
    Returns:
        str: The sanitized filename.
    """
    # Replace periods with underscores (except for file extensions)
    filename = str(filename)
    
    base_name = filename.replace('.', '_')
    
    # Keep only alphanumeric characters, underscores, and hyphens
    sanitized = ''.join(c for c in base_name if c.isalnum() or c == '_' or c == '-')
    
    # Replace multiple consecutive underscores with a single one
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    
    return sanitized

def process_scan(dcm_path: str, out_path: str, subject_name: str, session_name: str, scan_name: str, force_slice_thickness: bool = False, conversion_method: str = 'dcm2niix'):
    """Processes each scan, converting DICOM to NIfTI and transferring metadata."""
    os.makedirs(out_path, exist_ok=True)
    dcm_folder = os.path.dirname(dcm_path)
    
    # Sanitize the scan name for the output filename
    sanitized_scan_name = sanitize_filename(scan_name)
    sanitized_subject_name = sanitize_filename(subject_name)
    sanitized_session_name = sanitize_filename(session_name)
    
    # Use sanitized names for file output
    output_basename = f"{sanitized_subject_name}_{sanitized_session_name}_{sanitized_scan_name}"
    nifti_file = os.path.join(out_path, f"{output_basename}.nii.gz")
    json_file = os.path.join(out_path, f"{output_basename}.json")
    
    # Check if output files already exist, unless we're only transferring metadata
    if os.path.isfile(nifti_file) and os.path.isfile(json_file):
        print(f"\tOutput for {output_basename} already exists. Skipping...")
        
        # Transfer specified DICOM fields to the JSON sidecar
        transfer_dicom_fields_to_json(dcm_path, json_file, subject_name, session_name, scan_name, ['StudyDescription', 'SeriesDescription', 'AcquisitionDate', 'PatientAge', 'PatientWeight', 'PatientSex'])

        # For aMRI scans, transfer HeartRate to the JSON sidecar
        if scan_name == 'aMRI':
            transfer_dicom_fields_to_json(dcm_path, json_file, subject_name, session_name, scan_name, ['HeartRate'])

        return

    # Skip processing if cardiac images are missing for aMRI scans
    if scan_name == 'aMRI' and not check_cardiac_number_of_images(dcm_path):
        display_error(f"Cardiac frames missing for scan {output_basename}. Skipping...")
        return

    # Check for incorrect slice thickness and use a temporary directory if needed
    if scan_name == 'aMRI' and not incorrect_slice_thickness(dcm_path) or force_slice_thickness:
        use_temp_dir = True
        display_warning(f"    Incorrect slice thickness detected for {output_basename}. Adjusting in temporary folder.")
        dcm_folder = set_slice_thickness_temp(dcm_folder)
    elif '.zip:' in dcm_path:
        use_temp_dir = True
    else: 
        use_temp_dir = False
    
    # Convert DICOM to NIfTI using the specified method
    if conversion_method == 'mrconvert':
        cmd = [
            'mrconvert', dcm_folder, nifti_file
        ]
    else:  # Default to dcm2niix
        cmd = [
            'dcm2niix', '-z', 'y', '-b', 'y', '-w', '0',
            '-o', out_path, '-f', output_basename, dcm_folder
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for errors in the conversion process
    if result.returncode != 0:
        raise RuntimeError(f"{conversion_method} failed for scan: {dcm_folder} {result.stderr}")
    
    # Check if the outputs were successfully created
    if not os.path.isfile(nifti_file):
        raise RuntimeError(f"{conversion_method} failed to create scan: {dcm_folder}\n{result.stderr}")

    # Transfer specified DICOM fields to the JSON sidecar
    transfer_dicom_fields_to_json(dcm_path, json_file, subject_name, session_name, scan_name, ['StudyDescription', 'SeriesDescription', 'AcquisitionDate', 'PatientAge', 'PatientWeight', 'PatientSex'])

    # For aMRI scans, transfer HeartRate to the JSON sidecar
    if scan_name == 'aMRI':
        transfer_dicom_fields_to_json(dcm_path, json_file, subject_name, session_name, scan_name, ['HeartRate'])

    # Remove temporary directory if used
    if use_temp_dir:
        shutil.rmtree(dcm_folder)

def check_cardiac_number_of_images(dicom_file_path: str) -> bool:
    """
    Checks the DICOM field 'CardiacNumberOfImages' from the specified DICOM file.
    Returns True if the field is equal to 20, otherwise returns False and warns the user.
    
    Parameters:
    - dicom_file_path (str): Path to the DICOM file.
    
    Returns:
    - bool: True if 'CardiacNumberOfImages' is 20, False otherwise.
    """
    # Initialize output
    is_correct_number = False

    # Check if the file exists
    if not os.path.isfile(dicom_file_path):
        raise FileNotFoundError(f"DICOM file does not exist: {dicom_file_path}")

    try:
        # Read the DICOM file
        dicom_data = pydicom.dcmread(dicom_file_path)
        
        # Check for the 'CardiacNumberOfImages' field
        if hasattr(dicom_data, 'CardiacNumberOfImages'):
            cardiac_number_of_images = dicom_data.CardiacNumberOfImages
            if cardiac_number_of_images == 20:
                is_correct_number = True
            else:
                print(f"CardiacNumberOfImages is {cardiac_number_of_images}, expected 20.")
        else:
            print(f"'CardiacNumberOfImages' field not found in DICOM file: {dicom_file_path}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM or extract CardiacNumberOfImages: {str(e)}")

    return is_correct_number

def transfer_dicom_fields_to_json(dicom_file_path: str, json_path: str, subject_name: str, session_name: str, scan_name: str, dicom_fields: List[str]):
    """
    Extract specified DICOM fields from files in a scan directory and update
    the corresponding BIDS JSON sidecar file with these fields.
    
    Parameters:
    - scan_dir (str): Path to the scan directory containing DICOM files.
    - bids_output_dir (str): Path to the BIDS output directory.
    - subject_name (str): BIDS subject identifier (e.g., 'sub-01').
    - session_name (str): BIDS session identifier (e.g., 'ses-01').
    - scan_name (str): BIDS scan identifier (e.g., 'T1w').
    - dicom_fields (List[str]): List of DICOM fields to extract and transfer.
    """
    # Ensure the scan directory exists
    if not os.path.isfile(dicom_file_path):
        raise FileNotFoundError(f"Dicom does not exist: {dicom_file_path}")

    # Extract specified fields from the first DICOM file in the scan directory
    dicom_data = extract_dicom_fields(dicom_file_path, dicom_fields)
    
    # Update the JSON sidecar with the extracted DICOM fields
    update_json_sidecar(json_path, dicom_data)

def extract_dicom_fields(dicom_file_path: str, dicom_fields: List[str]) -> Dict[str, str]:
    """
    Extract specified DICOM fields from the first DICOM file in a scan directory.
    
    Parameters:
    - dicom_file_path (str): Path to the DICOM.
    - dicom_fields (List[str]): List of DICOM fields to extract.
    
    Returns:
    - Dict[str, str]: Dictionary containing extracted DICOM fields.
    """
    dicom_data = {}
    try:
        info = pydicom.dcmread(dicom_file_path, force=True)
        for field in dicom_fields:
            if hasattr(info, field):
                dicom_data[field] = getattr(info, field)
            else:
                print(f"DICOM field {field} not found in file {dicom_file_path}.")
                dicom_data[field] = "Not found"
    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM or extract fields due to: {str(e)}")
    
    return dicom_data

def update_json_sidecar(json_path: str, dicom_data: Dict[str, str]):
    """
    Update a JSON sidecar file with specified DICOM fields.
    
    Parameters:
    - json_path (str): Path to the JSON sidecar file.
    - dicom_data (Dict[str, str]): Dictionary containing DICOM fields.
    """
    # Check if the JSON file exists
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"The specified JSON file was not found: {json_path}")
    
    # Read JSON file
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file at {json_path} due to: {str(e)}")
    
    # Update JSON data with DICOM fields
    json_data.update(dicom_data)
    
    # Write updated JSON data back to file
    try:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"Failed to update JSON file at {json_path} due to: {str(e)}")

def incorrect_slice_thickness(dicom_file_path: str) -> bool:
    """Check if the SliceThickness to SpacingBetweenSlices ratio in the first DICOM file is >= 2."""
    dicom_data = pydicom.dcmread(dicom_file_path)
    
    slice_thickness = getattr(dicom_data, "SliceThickness", None)
    spacing_between_slices = getattr(dicom_data, "SpacingBetweenSlices", None)
    
    if slice_thickness is None or spacing_between_slices is None:
        raise ValueError(f"Missing SliceThickness or SpacingBetweenSlices in file: {dicom_file_path}")
    
    incorrect_thickness = (slice_thickness / spacing_between_slices) >= 1.95
    return incorrect_thickness

def set_slice_thickness_temp(scan_dir: str) -> str:
    """
    Sets the slice thickness in all DICOM files in `scan_dir` to the value of `SpacingBetweenSlices`,
    saving them in a temporary directory.

    Parameters:
    - scan_dir (str): Path to the folder containing the original DICOM files.

    Returns:
    - str: Path to the temporary folder containing modified DICOM files.
    """
    temp_dir = tempfile.mkdtemp()

    for root, _, files in os.walk(scan_dir):
        for filename in files:
            if filename.endswith('.dcm'):
                dicom_path = os.path.join(root, filename)
                try:
                    # Read the DICOM file
                    dicom_data = pydicom.dcmread(dicom_path)

                    # Check if 'SpacingBetweenSlices' is available
                    if hasattr(dicom_data, 'SpacingBetweenSlices'):
                        # Set 'SliceThickness' to 'SpacingBetweenSlices'
                        dicom_data.SliceThickness = dicom_data.SpacingBetweenSlices
                    else:
                        display_warning(f"Warning: 'SpacingBetweenSlices' not found in {filename}. Skipping file.")
                        continue  # Skip this file if 'SpacingBetweenSlices' is missing

                    # Save the modified file to the temporary directory
                    output_path = os.path.join(temp_dir, filename)
                    dicom_data.save_as(output_path)
                
                except Exception as e:
                    display_error(f"Failed to process {filename}: {e}")

    return temp_dir

def display_warning(message):
    """Prints a warning message in yellow."""
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    print(f"{YELLOW}Warning: {message}{RESET}")


def display_error(message):
    """Prints an error message in red."""
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}Error: {message}{RESET}")

def setup_sheets(bids_dir: str) -> None:
    """
    Main function to generate participants TSV and JSON files for a BIDS dataset.

    Parameters:
    - bids_dir: str
        Path to the BIDS directory containing raw data and JSON sidecar files.
    - file_extension: str
        File extension to look for when processing JSON sidecar files (default is '.json').
    """
    create_participants_tsv(bids_dir)
    create_participants_json(bids_dir)


def create_participants_tsv(bids_dir: str) -> None:
    """
    Creates a participants.tsv file with one row per subject-session, summarizing extracted fields.


    Parameters:
    - bids_dir: str
        Path to the BIDS directory containing raw data and JSON sidecar files.
    - file_extension: str
        File extension to look for when processing JSON sidecar files (default is '.json').
    """
    # TSV path and name
    tsv_path = os.path.join(bids_dir, 'participants.tsv')

    # Ensure BIDS directory exists
    if not os.path.isdir(bids_dir):
        raise ValueError(f"The specified BIDS directory does not exist: {bids_dir}")

    # Fields to extract from JSON files
    fields_to_extract = ['StudyDescription', 'HeartRate', 'AcquisitionDate', 
                         'AcquisitionTime', 'PatientAge', 'PatientWeight', 'PatientSex']

    # Define the header for the TSV
    tsv_data = [['Subject', 'Session', 'StudyDescription', 
                 'AcquisitionDate', 'AcquisitionTime', 'PatientAge', 
                 'PatientWeight', 'PatientSex', 'HeartRate']]

    # Extract subject, session, and scan information
    subjects_sessions_scans_bids = list_bids_subjects_sessions_scans(data_directory=bids_dir, file_extension='.json')

    # Dictionary to store consolidated data for each subject-session
    consolidated_data = {}

    # Iterate over subjects, sessions, and scans
    for subject_name, sessions in subjects_sessions_scans_bids.items():
        for session_name, scans in sessions.items():
            # Initialize a dictionary for this subject-session
            key = (subject_name, session_name)
            if key not in consolidated_data:
                consolidated_data[key] = {field: 'NA' for field in fields_to_extract}

            for scan_type, file_paths in scans.items():
                for json_path in file_paths.values():
                    try:
                        attributes = extract_fields(json_path, fields_to_extract, warn=False)
                    except ValueError as e:
                        print(f"Warning: {e}")
                        continue

                    # Update consolidated data for this subject-session
                    for field in fields_to_extract:
                        if attributes.get(field) and consolidated_data[key][field] == 'NA':
                            consolidated_data[key][field] = attributes[field]

    # Populate the TSV data from consolidated results
    for (subject_name, session_name), attributes in consolidated_data.items():
        tsv_data.append([
            subject_name,
            session_name,
            attributes.get('StudyDescription', 'NA'),
            attributes.get('AcquisitionDate', 'NA'),
            attributes.get('AcquisitionTime', 'NA'),
            attributes.get('PatientAge', 'NA'),
            attributes.get('PatientWeight', 'NA'),
            attributes.get('PatientSex', 'NA'),
            attributes.get('HeartRate', 'NA')
        ])

    # Write to TSV
    with open(tsv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(tsv_data)
    print(f"TSV file saved to {tsv_path}")


def create_participants_json(bids_dir: str) -> None:
    """
    Creates a participants.json file describing the levels of the participants.tsv file.

    Parameters:
    - bids_dir: str
        Path to the BIDS directory containing the participants.tsv file.
    """
    # JSON path and name
    json_path = os.path.join(bids_dir, 'participants.json')

    # Description of TSV levels
    json_data = {
        "StudyDescription": {
            "Description": "Description of the study from the imaging dataset"
        },
        "Subject": {
            "Description": "Participant ID"
        },
        "Session": {
            "Description": "Session ID"
        },
        "Scan": {
            "Description": "Scan name extracted from the file name"
        },
        "AcquisitionDate": {
            "Description": "Date of the acquisition"
        },
        "AcquisitionTime": {
            "Description": "Time of the acquisition"
        },
        "PatientAge": {
            "Description": "Age of the participant at the time of the scan"
        },
        "PatientWeight": {
            "Description": "Weight of the participant in kilograms"
        },
        "PatientSex": {
            "Description": "Sex of the participant"
        },
        "HeartRate": {
            "Description": "Heart rate during the scan"
        }
    }

    # Write JSON data to file
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON file saved to {json_path}")

def extract_fields(json_path: str, fields_to_extract: List[str], warn: bool = True) -> Dict[str, str]:
    """
    Extracts specified fields from a JSON file.

    Parameters:
    - json_path: str
        Path to the JSON file.
    - fields_to_extract: List[str]
        List of fields to extract from the JSON file.

    Returns:
    - Dict[str, str]:
        Dictionary with extracted fields and their values. If a field is missing,
        it will have the value 'NA'.
    """
    attributes = {}

    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read JSON file at {json_path} due to: {e}")

    # Extract each field, set 'NA' if the field is missing
    for field in fields_to_extract:
        attributes[field] = str(json_data.get(field, 'NA'))
        if field not in json_data and warn:
            display_warning(f'Missing field "{field}" in JSON file: {json_path}')
    return attributes

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description=(
            "dcm2bids: Convert DICOM files to a BIDS-compliant* structure.\n\n"
            "This script processes DICOM files, converts them to NIfTI format, "
            "and organizes them into a BIDS-compliant* directory structure. It also "
            "transfers relevant metadata to JSON sidecars. *File naming conventions "
            "may not fully adhere to bids conventions in current version "
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "data_directory",
        type=str,
        help="Path to the input directory containing all DICOM files."
    )
    parser.add_argument(
        "bids_output_dir",
        type=str,
        help="Path to the output BIDS directory."
    )

    # Optional arguments
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Enable searching for and processing DICOM files within ZIP archives."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging output."
    )
    parser.add_argument(
        "-c", "--conversion-method",
        type=str,
        choices=['dcm2niix', 'mrconvert'],
        default='dcm2niix',
        help="Specify the DICOM to NIfTI conversion method to use."
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Call the main function
    dcm2bids(
        data_directory=args.data_directory,
        bids_output_dir=args.bids_output_dir,
        zip=args.zip,
        conversion_method=args.conversion_method
    )

if __name__ == "__main__":
    main()