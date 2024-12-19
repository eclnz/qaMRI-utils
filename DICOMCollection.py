from dataclasses import dataclass, fields
from datetime import datetime
import pydicom
import nibabel as nib
from typing import Dict, Optional, Union, BinaryIO
import zipfile
import io
import os

def sanitize_string(string: str) -> str:
    # Placeholder for sanitization logic
    return string.strip().lower()

@dataclass
class ScanInfo:
    pass

@dataclass
class DicomInfo(ScanInfo):
    study_description: str
    patient_name: str
    series_description: str
    acquisition_date: str
    cohort: str = ""
    subject_id: str = ""
    session: str = ""
    scan: str = ""

    @classmethod
    def from_dicom(cls, dicom: pydicom.Dataset) -> "DicomInfo":
        extracted_fields = {
            field.name: getattr(dicom, field.name.capitalize(), "")
            for field in fields(cls)
            if field.name not in {"cohort", "subject_id", "session", "scan"}
        }
        instance = cls(**extracted_fields)
        instance.parse_dicom_attributes()
        return instance

    def parse_dicom_attributes(self) -> None:
        self._parse_identifiers()
        self._sanitize_scan()
        self._parse_acquisition_date()

    def _parse_identifiers(self) -> None:
        def parse_parts(parts):
            cohort, subject_id, session = None, None, None
            for part in reversed(parts):
                if part.isdigit() and subject_id is None:
                    subject_id = str(part)
                elif part.isupper() and len(part) == 1 and session is None:
                    session = part
                elif part.isalpha() and cohort is None:
                    cohort = part.lower()
            return cohort, subject_id, session

        parts = self.study_description.split()
        self.cohort, self.subject_id, self.session = parse_parts(parts)

        if not (self.cohort and self.subject_id and self.session):
            parts = self.patient_name.split()
            self.cohort, self.subject_id, self.session = parse_parts(parts)

    def _sanitize_scan(self) -> None:
        self.scan = sanitize_string(self.series_description)

    def _parse_acquisition_date(self) -> None:
        try:
            self.acquisition_date = str(datetime.strptime(self.acquisition_date, "%Y%m%d").year)
        except (ValueError, TypeError):
            raise ValueError("Scan date in unsupported format")

@dataclass
class NiftiInfo(ScanInfo):
    shape: tuple
    affine: list

    @classmethod
    def from_nifti(cls, nifti: Union[nib.Nifti1Image, nib.Nifti2Image]) -> "NiftiInfo":
        if isinstance(nifti, (nib.Nifti1Image, nib.Nifti2Image)):
            return cls(
                shape=nifti.shape,
                affine=nifti.affine.tolist()
            )
        else:
            raise TypeError("Unsupported NIfTI image type")

class Scan:
    def __init__(
        self, 
        scan_id: str, 
        scan_info: Union[DicomInfo, NiftiInfo],
        scan_file: Union[nib.Nifti1Image, nib.Nifti2Image, pydicom.Dataset]
    ):
        self.scan_id = scan_id
        self.scan_info = scan_info
        self.scan_file = scan_file
        self.scan_type = type(scan_info).__name__  # Store the type of scan_info

    def get_data(self) -> Union[DicomInfo, NiftiInfo]:
        return self.scan_info

    def __repr__(self):
        return f"{self.__class__.__name__}(scan_id={self.scan_id}, info={self.scan_info})"

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.scans: Dict[str, Scan] = {}

    def add_scan(self, scan: Scan):
        self.scans[scan.scan_id] = scan

    def get_scan(self, scan_id: str) -> Optional[Scan]:
        return self.scans.get(scan_id)

    def __repr__(self):
        return f"Session(session_id={self.session_id}, scans={list(self.scans.keys())})"

class Subject:
    def __init__(self, subject_id: str):
        self.subject_id = subject_id
        self.sessions: Dict[str, Session] = {}

    def add_session(self, session: Session):
        self.sessions[session.session_id] = session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def __repr__(self):
        return f"Subject(subject_id={self.subject_id}, sessions={list(self.sessions.keys())})"
    
class MediaCollection:
    def __init__(self):
        self.subjects: Dict[str, Subject] = {}

    def add_subject(self, subject: Subject) -> None:
        self.subjects[subject.subject_id] = subject

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        return self.subjects.get(subject_id)

    def populate_from_folder(self, folder_path: str) -> None:
        """
        Populate the MediaCollection from a given folder, recursively traversing
        its structure to identify and parse DICOM and NIfTI files. For DICOM files,
        assume they are in folders without nested folders. Check validity by attempting
        to read 10 files and verifying the presence of StudyInstanceUID.
        
        Parameters:
            folder_path (str): Path to the folder to parse.
        """
        for root, dirs, files in os.walk(folder_path):
            if not dirs:  # Check if the folder contains no subfolders
                if self._validate_dicom_folder(root, files):
                    self._process_dicom_folder(root, files)
                elif self._validate_nifti_folder(root, files):
                    self._process_nifti_files(root, files)

    def _validate_dicom_folder(self, folder: str, files: list) -> bool:
        """
        Validate if a folder contains DICOM files by attempting to read up to 10 files
        and checking for the StudyInstanceUID attribute.

        Parameters:
            folder (str): Path to the folder.
            files (list): List of file names in the folder.

        Returns:
            bool: True if the folder contains valid DICOM files, False otherwise.
        """
        for file in files[:10]:  # Test up to 10 files
            file_path = os.path.join(folder, file)
            try:
                dicom_data = pydicom.dcmread(file_path, stop_before_pixels=True)
                if hasattr(dicom_data, "StudyInstanceUID"):
                    return True
            except Exception:
                continue
        return False

    def _process_dicom_folder(self, folder: str, files: list) -> None:
        """
        Process a folder containing DICOM files and add them to the MediaCollection.

        Parameters:
            folder (str): Path to the folder.
            files (list): List of file names in the folder.
        """
        for file in files:
            file_path = os.path.join(folder, file)
            
            try:
                dicom_data = pydicom.dcmread(file_path)
            except Exception as e:
                print(f"Error processing DICOM file {file_path}: {e}")
                continue
                
            # Standardised structure containing data.
            dicom_info = DicomInfo.from_dicom(dicom_data)

            subject = self._get_or_create_subject(dicom_info.subject_id)
            session = subject.get_session(dicom_info.session) or Session(dicom_info.session)
            scan = Scan(scan_id=file, scan_info=dicom_info, scan_file=dicom_data)

            session.add_scan(scan)
            subject.add_session(session)
            self.add_subject(subject)
                
    def _validate_nifti_folder(self, folder: str, files: list) -> bool:
        for file in files:
            file_path = os.path.join(folder, file)
            if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                return True
        return False

    def _process_nifti_files(self, folder: str, files: list) -> None:
        """
        Process NIfTI files in a folder and add them to the MediaCollection.

        Parameters:
            folder (str): Path to the folder.
            files (list): List of file names in the folder.
        """
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                file_path = os.path.join(folder, file)
                try:
                    nifti_data = nib.load(file_path)
                    nifti_info = NiftiInfo.from_nifti(nifti_data) #type:ignore

                    # Using folder structure to deduce subject/session IDs (example logic)
                    parts = folder.split(os.sep)
                    subject_id = parts[-2] if len(parts) > 1 else "unknown_subject"
                    session_id = parts[-1] if len(parts) > 0 else "unknown_session"

                    subject = self._get_or_create_subject(subject_id)
                    session = subject.get_session(session_id) or Session(session_id)
                    scan = Scan(scan_id=file, scan_info=nifti_info, scan_file=nifti_data)#type:ignore

                    session.add_scan(scan)
                    subject.add_session(session)
                    self.add_subject(subject)
                except Exception as e:
                    print(f"Error processing NIfTI file {file_path}: {e}")

    def _get_or_create_subject(self, subject_id: str) -> Subject:
        """
        Retrieve an existing subject or create a new one if it does not exist.

        Parameters:
            subject_id (str): Identifier for the subject.

        Returns:
            Subject: The existing or newly created subject.
        """
        if subject_id not in self.subjects:
            self.subjects[subject_id] = Subject(subject_id=subject_id)
        return self.subjects[subject_id]

    def __repr__(self) -> str:
        return f"MediaCollection(subjects={list(self.subjects.keys())})"
