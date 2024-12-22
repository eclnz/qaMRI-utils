from dataclasses import dataclass
from datetime import datetime
import pydicom
from typing import Dict, Optional, List
import os
import logging
import hashlib

@dataclass
class DicomInfo:
    """Represents DICOM metadata information.

    Attributes:
        StudyDescription (str, optional): Description of the study
        PatientName (str, optional): Name of the patient
        SeriesDescription (str, optional): Description of the series
        AcquisitionDate (str, optional): Date of acquisition in YYYYMMDD format
        dcm_path (str): Path to DICOM file
    """
    SCAN_CATEGORIES = {
        "anat": ["t1", "t2", "flair", "amri", "anatomy", "bravo", "inhance", "stir", "sag"],
        "dwi": ["dwi", "dki", "dsir", "dsires", "diffusion"],
        "func": ["rs_", "resting", "fmri", "default_mode", "attention_network", "visual_network"],
        "maps": ["map", "b0", "t1_map", "t2_map", "pd_map", "quantification"],
        "asl": ["easl", "cbf", "transit_corrected", "perfusion", "asl"],
        "fmaps": ["phasediff", "magnitude", "fieldmap", "b0"],
        "pet": ["pet", "suvr", "amyloid"],
    }

    StudyDescription: str = ""
    PatientName: str = ""
    SeriesDescription: str = ""
    AcquisitionDate: str = ""
    dcm_path: str = ""

    def __post_init__(self):
        """Initialize derived attributes."""
        self._parse_metadata()

    @classmethod
    def from_dicom(cls, dicom: pydicom.Dataset) -> "DicomInfo":
        """Create DicomInfo from DICOM dataset."""
        return cls(
            StudyDescription=str(getattr(dicom, "StudyDescription", "")),
            PatientName=str(getattr(dicom, "PatientName", "")),
            SeriesDescription=str(getattr(dicom, "SeriesDescription", "")),
            AcquisitionDate=str(getattr(dicom, "AcquisitionDate", "")),
        )

    def _parse_metadata(self) -> None:
        """Parse and validate metadata."""
        self.scan_id = self._generate_scan_id()
        self._parse_acquisition_date()
        self._parse_scan_group()

    def _generate_scan_id(self) -> str:
        """Generate a unique scan ID based on available metadata."""
        # Use all available metadata to create a unique identifier
        metadata_parts = []
        if self.StudyDescription:
            metadata_parts.append(self.StudyDescription)
        if self.PatientName:
            metadata_parts.append(self.PatientName)
        if self.SeriesDescription:
            metadata_parts.append(self.SeriesDescription)
        if self.AcquisitionDate:
            metadata_parts.append(self.AcquisitionDate)
        if self.dcm_path:  # Include file path as fallback for uniqueness
            metadata_parts.append(self.dcm_path)

        # If no metadata is available, use a timestamp
        if not metadata_parts:
            metadata_parts.append(str(datetime.now().timestamp()))
            
        identifier = "_".join(metadata_parts)
        hash_object = hashlib.md5(identifier.encode(), usedforsecurity=False)
        return hash_object.hexdigest()[:8]  # Use first 8 characters of hash

    def _parse_acquisition_date(self) -> None:
        """Parse and validate acquisition date if present."""
        if not self.AcquisitionDate:
            return

        if not isinstance(self.AcquisitionDate, str) or len(self.AcquisitionDate) != 8 or not self.AcquisitionDate.isdigit():
            logging.warning(f"AcquisitionDate '{self.AcquisitionDate}' is not in the expected format 'YYYYMMDD'")
            return

        try:
            self.acquisition_date = datetime.strptime(self.AcquisitionDate, "%Y%m%d")
        except ValueError as err:
            logging.warning(f"Failed to parse AcquisitionDate '{self.AcquisitionDate}': {err}")

    def _parse_scan_group(self) -> None:
        """Determine scan group based on series description if present."""
        self.scan_group = "other"  # Default category
        
        if not self.SeriesDescription:
            return
            
        scan_name_lower = self.SeriesDescription.lower()
        for category, keywords in self.SCAN_CATEGORIES.items():
            if any(keyword in scan_name_lower for keyword in keywords):
                self.scan_group = category
                break

class Scan:
    """Represents a single scan with its metadata."""
    def __init__(self, scan_id: str, dicom_info: Optional[DicomInfo] = None):
        self.scan_id = scan_id
        self.dicom_info = dicom_info

    def add_dicom_info(self, dicom_info: DicomInfo) -> None:
        self.dicom_info = dicom_info

    def __repr__(self):
        return f"Scan(scan_id={self.scan_id}, dicom_info={self.dicom_info})"

class DicomCollection:
    """Collection of DICOM scans."""
    def __init__(self):
        self.scans: Dict[str, Scan] = {}

    def add_scan(self, scan: Scan) -> None:
        """Add a scan to the collection."""
        if not isinstance(scan, Scan):
            raise TypeError("Expected Scan object")
        if not scan.scan_id:
            raise ValueError("Scan ID cannot be empty")
        self.scans[scan.scan_id] = scan

    def get_scan(self, scan_id: str) -> Optional[Scan]:
        """Retrieve a scan by its ID."""
        return self.scans.get(scan_id)

    def get_unique_scan_names(self) -> List[str]:
        """Get list of unique scan names."""
        unique_names = set()
        for scan in self.scans.values():
            if scan.dicom_info and scan.dicom_info.SeriesDescription:
                unique_names.add(scan.dicom_info.SeriesDescription)
        return sorted(unique_names)

    def populate_from_folder(self, folder_path: str) -> None:
        """Populate collection from a folder of DICOM files."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.dcm'):
                    file_path = os.path.join(root, file)
                    try:
                        dicom_data = pydicom.dcmread(file_path)
                        dicom_info = DicomInfo.from_dicom(dicom_data)
                        dicom_info.dcm_path = file_path
                        scan = Scan(scan_id=dicom_info.scan_id, dicom_info=dicom_info)
                        self.add_scan(scan)
                    except Exception as err:
                        logging.error(f"Failed to process DICOM file {file_path}: {err}")