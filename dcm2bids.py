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

def dcm2bids(data_directory: str, bids_output_dir: str, year: str, transfer: bool = False, zip: bool = False):
    bids_raw_output_dir = os.path.join(bids_output_dir, 'raw')
    
    # List all subjects, sessions, and scans containing the specified file type
    subjects_sessions_scans = list_non_bids_subjects_sessions_scans(data_directory, zip)

    # Build the series structure and get unique series descriptions
    unique_series = build_series_list(subjects_sessions_scans)

    # Display dropdown menu to select series descriptions
    selected_series = display_dropdown_menu(unique_series, title_text="Select scans to add to BIDS")
    
    if len(selected_series) == 0:
        display_error("No scans found or selected")
        return None

    print("Selected series descriptions for processing:")
    for series in selected_series:
        print(f" - {series}")
        
    # Process only scans that match the selected series descriptions
    for subject_id, sessions in subjects_sessions_scans.items():
        # Include cohort in subject name 
        cohort = next(iter(next(iter(sessions.values())).values())).get("cohort", "unknown") if sessions else "unknown"
        subject_name = f"sub-{cohort}{subject_id}"
        print(f"Processing Subject: {subject_name}")

        for session_id, scans in sessions.items():
            # Extract the year from the session metadata
            year = scans[next(iter(scans))]["date"] if scans else "unknown"
            session_name = f"ses-{year}{session_id}"
            print(f"  Processing Session: {session_name}")

            for scan, metadata in scans.items():
                series_description = scan  # The scan key represents the series description

                # Skip scans that are not in the selected series
                if series_description not in selected_series:
                    continue

                raw_path = metadata["dicom_path"]

                try:
                    # Standardize the scan name based on series description
                    standardized_scan_name = standardize_scan_name(series_description)
                    out_path = os.path.join(bids_raw_output_dir, subject_name, session_name, 'anat')

                    # Check if the raw path is from a ZIP
                    if '.zip:' in raw_path:
                        # Split the ZIP path and internal file path
                        zip_path, internal_path = raw_path.split('.zip:')
                        zip_path += '.zip'  # Add back the ".zip" extension

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
                        standardized_scan_name, bids_raw_output_dir, transfer
                    )
                    
                except Exception as e:
                    display_error(f"    Failed to process scan {scan}. Error: {e}")

                finally:
                    # Clean up the temporary directory if it was created
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

    create_bids_csv(bids_output_dir)

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

def build_series_list(subjects_sessions_scans: DefaultDict[str, DefaultDict[str, DefaultDict[str, DefaultDict[str, str]]]]) -> list:
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

def list_bids_subjects_sessions_scans(data_directory: str, file_extension: str) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
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
        for entry in path.iterdir():
            if entry.is_dir():
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

            elif entry.is_file() and entry.name.endswith(file_extension):
                # Extract metadata
                parent_session = entry.parent.name
                parent_subject = entry.parent.parent.name

                if not parent_subject.startswith("sub-") or not parent_session.startswith("ses-"):
                    continue  # Ignore files not in valid subject/session directories

                # Extract scan description
                parts = entry.name.split("_desc-")
                if len(parts) > 1:
                    scan = parts[1]
                else:
                    scan = "unknown"

                # Populate the structure
                subjects_sessions_scans[parent_subject][parent_session][scan]["scan_path"] = str(entry)

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
) -> DefaultDict[str, DefaultDict[str, DefaultDict[str, DefaultDict[str, str]]]]:
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
                "SeriesDescription": getattr(dicom, "SeriesDescription", "" ),
                "AcquisitionDate": getattr(dicom, "AcquisitionDate", ""),
            }

            # Split Description into components
            parts = dicom_info["StudyDescription"].split("_")
            if len(parts) < 5:
                display_warning(f"Invalid SeriesDescription format in {path_descriptor}: {dicom_info['StudyDescription']}")
                return True

            cohort = parts[2].lower()
            subject_id = parts[-2]
            session_id = parts[-1]
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
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # Get all file entries in the ZIP
                entries = zip_file.namelist()

                # Track folders that have already been processed
                processed_folders = set()
                found_dicom = False

                for file_name in entries:
                    # Skip directories
                    if file_name.endswith('/'):
                        continue

                    # Determine the folder of the current file
                    folder = '/'.join(file_name.split('/')[:-1])

                    # Skip this file if its folder has already been processed
                    if folder in processed_folders:
                        continue

                    # Check if the file is a valid DICOM
                    with zip_file.open(file_name) as file:
                        if is_valid_dicom(file):
                            # Process the DICOM
                            if process_dicom(file, path_descriptor=f"{zip_path}:{file_name}"): #type: ignore
                                # Mark this folder as processed
                                processed_folders.add(folder)
                                found_dicom = True

                return found_dicom

        except Exception as e:
            print(f"Error processing ZIP file {zip_path}: {e}")
            return False

    def process_file(entry, zip):
        """
        Process a single file entry, either as a DICOM or a ZIP file.
        """
        if entry.name.endswith('.zip') and zip:
            process_zip(entry.path)
            return False # Dont want to stop looking in folder containing zip because it may have other important things.
        if is_valid_dicom(entry.path):
            return process_dicom(entry.path, path_descriptor=entry.path)
        return False

    def recursive_search(folder_path: str, zip: bool) -> bool:
        """
        Recursively searches for DICOM files in the directory and organizes them.
        """
        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.name.startswith('.'):
                        # Skip hidden files or directories
                        continue

                    if entry.is_dir():
                        if recursive_search(entry.path, zip):
                            continue

                    elif entry.is_file():
                        if process_file(entry, zip):
                            return True

        except Exception as e:
            print(f"Error accessing folder {folder_path}: {e}")
        
        return False

    # Start recursive search from the root directory
    recursive_search(data_directory, zip)

    # Convert defaultdict to standard dictionary for return
    return subjects_sessions_scans

def display_dropdown_menu(str_list: list[str], title_text: str):
    """Display a dropdown menu to select multiple series descriptions using arrow keys, Space to select, and Enter to confirm."""
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
                return [str_list[i] for i in selected_items]  # Return the selected series descriptions

    selected_series = curses.wrapper(navigate_menu)
    return selected_series

def standardize_scan_name(series_description: str) -> str:
    """Standardizes scan name based on SeriesDescription."""
    series_description = series_description.lower()
    if 'amri' in series_description:
        return 'aMRI'
    elif 't1' in series_description:
        return 'T1'
    elif 't2' in series_description:
        return 'T2'
    elif 'dwi' in series_description:
        return 'dwi'
    else:
        return series_description

def process_scan(dcm_path: str, out_path: str, subject_name: str, session_name: str, scan_name: str, bids_output_dir: str, transfer: bool):
    """Processes each scan, converting DICOM to NIfTI and transferring metadata."""
    os.makedirs(out_path, exist_ok=True)
    dcm_folder = os.path.join(dcm_path,'..')
    
    nifti_file = os.path.join(out_path, f"{subject_name}_{session_name}_{scan_name}.nii.gz")
    json_file = os.path.join(out_path, f"{subject_name}_{session_name}_{scan_name}.json")

    # Check if the file exists
    if os.path.isfile(nifti_file):
        print("\tSubject already processed. Skipping...")
        return

    # Skip processing if cardiac images are missing for aMRI scans
    if scan_name == 'aMRI' and not check_cardiac_number_of_images(dcm_path):
        display_error(f"Cardiac frames missing for scan {subject_name}_{session_name}_{scan_name}. Skipping...")
        return

    # Check for incorrect slice thickness and use a temporary directory if needed
    if scan_name == 'aMRI' and not check_slice_thickness(dcm_path):
        use_temp_dir = True
        display_warning(f"    Incorrect slice thickness detected for {subject_name}_{session_name}_{scan_name}. Adjusting in temporary folder.")
        dcm_folder = set_slice_thickness_temp(dcm_folder)
    else: 
        use_temp_dir = False

    # Check if output files already exist, unless we’re only transferring metadata
    if os.path.isfile(nifti_file) and os.path.isfile(json_file) and not transfer:
        print(f"Output for {subject_name}_{session_name}_{scan_name} already exists. Skipping...")
    else:
        # Convert DICOM to NIfTI unless we're only transferring metadata
        if not transfer:
            cmd = [
                'dcm2niix', '-z', 'y', '-b', 'y', '-w', '0',
                '-o', out_path, '-f', f"{subject_name}_{session_name}_{scan_name}", dcm_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"dcm2niix failed for scan: {dcm_path}\n{result.stderr}")

        # Transfer specified DICOM fields to the JSON sidecar
        transfer_dicom_fields_to_json(dcm_path, bids_output_dir, subject_name, session_name, scan_name, ['StudyDescription', 'SeriesDescription', 'AcquisitionDate', 'PatientAge', 'PatientWeight', 'PatientSex'])

        # For aMRI scans, transfer HeartRate to the JSON sidecar
        if scan_name == 'aMRI':
            transfer_dicom_fields_to_json(dcm_path, bids_output_dir, subject_name, session_name, scan_name, ['HeartRate'])

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

def transfer_dicom_fields_to_json(dicom_file_path: str, bids_output_dir: str, subject_name: str, session_name: str, scan_name: str, dicom_fields: List[str]):
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
    
    # Define the path to the expected JSON sidecar file
    json_path = os.path.join(bids_output_dir, subject_name, session_name, 'anat',
                             f"{subject_name}_{session_name}_{scan_name}.json")

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

def check_slice_thickness(dicom_file_path: str) -> bool:
    """Check if the SliceThickness to SpacingBetweenSlices ratio in the first DICOM file is >= 2."""
    dicom_data = pydicom.dcmread(dicom_file_path)
    
    slice_thickness = getattr(dicom_data, "SliceThickness", None)
    spacing_between_slices = getattr(dicom_data, "SpacingBetweenSlices", None)
    
    if slice_thickness is None or spacing_between_slices is None:
        raise ValueError(f"Missing SliceThickness or SpacingBetweenSlices in file: {dicom_file_path}")
    
    return (slice_thickness / spacing_between_slices) >= 1.95

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

def create_bids_csv(bids_dir: str, 
                    subjects_sessions_scans: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None, 
                    file_extension: str = '.json', 
                    bids: bool = True) -> None:
    """
    Scans a BIDS directory or uses a precomputed dictionary, extracts specified fields from JSON sidecar files,
    and writes the data to a CSV file.

    Parameters:
    - bids_dir: str
        Path to the BIDS directory containing raw data and JSON sidecar files.
    - file_extension: str
        The file extension to look for in each directory, typically '.json'.
    - bids: bool
        If True, assumes scan files are within the session folder directly.
    """
    # CSV path and name
    csv_path = os.path.join(bids_dir, 'participants.csv')

    # Check if BIDS directory exists
    if not os.path.isdir(bids_dir):
        raise ValueError(f"The specified BIDS directory does not exist: {bids_dir}")

    # Fields to extract from JSON files
    fields_to_extract = ['StudyDescription', 'SeriesDescription', 'HeartRate', 'AcquisitionDate', 
                         'AcquisitionTime', 'PatientAge', 'PatientWeight', 'PatientSex']

    # Define the header for the CSV
    csv_data = [['StudyDescription', 'SeriesDescription', 'Subject', 'Session', 'ScanType', 'Scan', 
                 'AcquisitionDate', 'AcquisitionTime', 'PatientAge', 'PatientWeight', 'PatientSex', 'HeartRate']]
    
    # Define the path to the raw directory
    raw_dir = os.path.join(bids_dir, 'raw')

    # If subjects_sessions_scans is not provided, generate it
    subjects_sessions_scans_bids = list_bids_subjects_sessions_scans(data_directory=raw_dir, file_extension=file_extension)

    # Iterate over each subject, session, and scan in subjects_sessions_scans
    for subject_name, sessions in subjects_sessions_scans_bids.items():
        for session_name, scans in sessions.items():
            for scan_type, file_paths in scans.items():
                for json_path in file_paths:
                    # Only process JSON files
                    if not json_path.endswith('.json'):
                        continue

                    # Extract scan name from the JSON file path
                    scan_name = os.path.basename(json_path).replace('.json', '')

                    # Try to extract fields and handle errors by filling with 'NA'
                    try:
                        attributes = extract_fields(json_path, fields_to_extract, warn=False)
                    except ValueError as e:
                        print(f"Warning: {e}")
                        continue  # Skip this file if there's an error

                    # Append extracted data to csv_data
                    csv_data.append([
                        attributes.get('StudyDescription', 'NA'),
                        attributes.get('SeriesDescription', 'NA'),
                        subject_name, 
                        session_name, 
                        scan_type, 
                        scan_name,
                        attributes.get('AcquisitionDate', 'NA'),
                        attributes.get('AcquisitionTime', 'NA'),
                        attributes.get('PatientAge', 'NA'),
                        attributes.get('PatientWeight', 'NA'),
                        attributes.get('PatientSex', 'NA'),
                        attributes.get('HeartRate', 'NA')
                    ])

    # Write the CSV data to a file
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    print(f"CSV file saved to {csv_path}")

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

# if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DICOM files to BIDS-compliant NIfTI format")
    parser.add_argument("data_directory", help="Path to the directory containing subject folders with DICOM files.")
    parser.add_argument("bids_output_dir", help="Path to the BIDS output directory.")
    parser.add_argument("--transfer", action="store_true", help="Only transfer DICOM attributes to .json sidecar without DICOM-to-NIfTI conversion.")
    parser.add_argument("--zip", action="store_true", help="Enable searching through zip files in addition to folders.")

    args = parser.parse_args()

    dcm2bids(args.data_directory, args.bids_output_dir, args.transfer, args.zip)

# # For debugging purposes: Manually sets folder.
# def main():
#     # Manually specify the variables
#     data_directory = "../qaMRI-clone/testData/raw_r"  # Replace with the actual path
#     bids_output_dir = "../qaMRI-clone/testData/BIDS7"  # Replace with the actual path
#     transfer = False  # Set to True to only transfer DICOM attributes without conversion
#     zip = False
    
#     # Run the function
#     dcm2bids(data_directory, bids_output_dir, transfer, zip)

# if __name__ == "__main__":
#     main()