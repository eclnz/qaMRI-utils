import os
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import tempfile

# Predefined test data
TEST_DICOM_FILES = [
    {
        "name": "file1.dcm",
        "type": "dicom",
        "acquisition_date": "20220101",
        "study_description": "ADNI_123_A",
        "patient_name": "Test_Subject_01",
        "series_description": "T1_MPRAGE"
    },
    {
        "name": "file2.dcm",
        "type": "dicom",
        "acquisition_date": "20220101",
        "study_description": "ADNI_123_A",
        "patient_name": "Test_Subject_01",
        "series_description": "DWI_b1000"
    }
]

def create_temp_dicom(filepath, acquisition_date="20220101", study_description="Study 1", 
                     patient_name="Test_Subject", series_description="Series 1"):
    """Helper function to create a mock DICOM file with valid headers."""
    # Create a minimal dataset
    ds = Dataset()
    ds.PatientName = patient_name
    ds.StudyDescription = study_description
    ds.SeriesDescription = series_description
    ds.StudyInstanceUID = generate_uid()
    ds.AcquisitionDate = acquisition_date

    # Add required file meta information
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Save the file
    pydicom.dcmwrite(filepath, ds, enforce_file_format=True)

def create_temp_nifti(filepath, shape=(64, 64, 64), affine=None):
    """Helper function to create a mock NIfTI file."""
    if affine is None:
        affine = np.eye(4)
    data = np.random.rand(*shape)  # Create random data
    nifti_image = nib.Nifti1Image(data, affine)
    nib.save(nifti_image, filepath)


def create_temp_files(file_descriptions, directory=None):
    """Create multiple files in a temporary directory, each DICOM in its own subfolder."""
    if directory is None:
        directory = tempfile.mkdtemp()

    os.makedirs(directory, exist_ok=True)
    file_paths = []

    for desc in file_descriptions:
        filename = desc["name"]
        filetype = desc["type"]

        if filetype == "dicom":
            # Create a subfolder for each DICOM using series description
            subfolder = os.path.join(directory, desc.get("series_description", "series").replace(" ", "_"))
            os.makedirs(subfolder, exist_ok=True)
            filepath = os.path.join(subfolder, filename)

            create_temp_dicom(
                filepath,
                acquisition_date=desc.get("acquisition_date", "20220101"),
                study_description=desc.get("study_description", "Study 1"),
                patient_name=desc.get("patient_name", "Test_Subject"),
                series_description=desc.get("series_description", "Series 1")
            )
        else:  # nifti files stay in main directory
            filepath = os.path.join(directory, filename)
            create_temp_nifti(filepath, shape=desc.get("shape", (64, 64, 64)), affine=desc.get("affine"))

        file_paths.append(filepath)

    return file_paths, directory


file_paths, test_dir = create_temp_files(file_descriptions=TEST_DICOM_FILES, directory=None)

from DICOMCollection import DicomCollection 
file_paths, test_dir = create_temp_files(TEST_DICOM_FILES, test_dir)
collection = DicomCollection()
collection.populate_from_folder(test_dir)