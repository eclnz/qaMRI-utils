import unittest
from DICOMCollection import DicomCollection, Scan, DicomInfo
from synthetic_data import create_temp_files, TEST_DICOM_FILES
from exemplar_data import download_test_dicom
import tempfile
import os
import shutil
import warnings

class TestDicomInfo(unittest.TestCase):
    """Test suite for DicomInfo class."""

    def setUp(self):
        self.dicom_info = DicomInfo(
            StudyDescription="Study_01",
            PatientName="Patient_01",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )

    def test_empty_fields(self):
        """Test that DicomInfo can be created with empty fields."""
        # Test with all empty fields
        dicom_info = DicomInfo()
        self.assertIsInstance(dicom_info, DicomInfo)
        self.assertTrue(hasattr(dicom_info, 'scan_id'))
        self.assertEqual(dicom_info.scan_group, "other")

        # Test with some empty fields
        dicom_info = DicomInfo(SeriesDescription="T1_MPRAGE")
        self.assertIsInstance(dicom_info, DicomInfo)
        self.assertEqual(dicom_info.scan_group, "anat")

    def test_scan_id_generation(self):
        """Test unique scan ID generation with various metadata combinations."""
        # Same complete metadata should generate same ID
        dicom_info2 = DicomInfo(
            StudyDescription="Study_01",
            PatientName="Patient_01",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )
        self.assertEqual(self.dicom_info.scan_id, dicom_info2.scan_id)

        # Different metadata should generate different ID
        dicom_info3 = DicomInfo(
            StudyDescription="Study_02",
            PatientName="Patient_01",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )
        self.assertNotEqual(self.dicom_info.scan_id, dicom_info3.scan_id)

        # Test ID generation with minimal metadata
        dicom_info4 = DicomInfo(SeriesDescription="T1_MPRAGE")
        self.assertTrue(dicom_info4.scan_id)  # Should still generate an ID
        self.assertNotEqual(dicom_info4.scan_id, self.dicom_info.scan_id)

        # Test ID generation with only file path
        dicom_info5 = DicomInfo(dcm_path="/path/to/file.dcm")
        self.assertTrue(dicom_info5.scan_id)
        self.assertNotEqual(dicom_info5.scan_id, dicom_info4.scan_id)

    def test_parse_acquisition_date(self):
        """Test parsing of acquisition date with various formats."""
        # Test valid date
        self.assertEqual(self.dicom_info.acquisition_date.year, 2022)
        self.assertEqual(self.dicom_info.acquisition_date.month, 1)
        self.assertEqual(self.dicom_info.acquisition_date.day, 1)

        # Test invalid date format - should not raise exception
        dicom_info = DicomInfo(AcquisitionDate="2022-01-01")
        self.assertFalse(hasattr(dicom_info, 'acquisition_date'))

        # Test invalid date parsing - should not raise exception
        dicom_info = DicomInfo(AcquisitionDate="00000000")
        self.assertFalse(hasattr(dicom_info, 'acquisition_date'))

        # Test missing date
        dicom_info = DicomInfo()
        self.assertFalse(hasattr(dicom_info, 'acquisition_date'))

    def test_parse_scan_group(self):
        """Test scan group categorization with various inputs."""
        # Test anatomical scan
        self.assertEqual(self.dicom_info.scan_group, "anat")

        # Test diffusion scan
        dwi_info = DicomInfo(SeriesDescription="DWI_b1000")
        self.assertEqual(dwi_info.scan_group, "dwi")

        # Test unknown category
        other_info = DicomInfo(SeriesDescription="Unknown_Sequence")
        self.assertEqual(other_info.scan_group, "other")

        # Test empty series description
        empty_info = DicomInfo()
        self.assertEqual(empty_info.scan_group, "other")

    def test_from_dicom(self):
        """Test creation from DICOM dataset with various field combinations."""
        import pydicom

        test_file, temp_dir = download_test_dicom()
        try:
            # Read the DICOM file
            ds = pydicom.dcmread(test_file, force=True)

            # Test with complete dataset
            dicom_info = DicomInfo.from_dicom(ds)
            self.assertIsInstance(dicom_info, DicomInfo)

            # Test with missing fields
            # Remove some attributes to test robustness
            if hasattr(ds, 'StudyDescription'):
                delattr(ds, 'StudyDescription')
            if hasattr(ds, 'SeriesDescription'):
                delattr(ds, 'SeriesDescription')

            dicom_info = DicomInfo.from_dicom(ds)
            self.assertIsInstance(dicom_info, DicomInfo)
            self.assertTrue(dicom_info.scan_id)  # Should still generate an ID
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

class TestScan(unittest.TestCase):
    """Test suite for Scan class."""

    def setUp(self):
        self.dicom_info = DicomInfo(
            StudyDescription="Study_01",
            PatientName="Patient_01",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )
        self.scan = Scan(scan_id=self.dicom_info.scan_id)

    def test_add_dicom_info(self):
        """Test adding DICOM information to a scan."""
        # Test with complete DicomInfo
        self.scan.add_dicom_info(self.dicom_info)
        self.assertEqual(self.scan.dicom_info, self.dicom_info)

        # Test with minimal DicomInfo
        minimal_info = DicomInfo(SeriesDescription="T1_MPRAGE")
        scan = Scan(scan_id=minimal_info.scan_id)
        scan.add_dicom_info(minimal_info)
        self.assertEqual(scan.dicom_info, minimal_info)

    def test_scan_representation(self):
        """Test string representation of scan."""
        # Test with complete DicomInfo
        self.scan.add_dicom_info(self.dicom_info)
        self.assertIn(self.dicom_info.scan_id, repr(self.scan))
        self.assertIn("dicom_info", repr(self.scan))

        # Test with minimal DicomInfo
        minimal_info = DicomInfo(SeriesDescription="T1_MPRAGE")
        scan = Scan(scan_id=minimal_info.scan_id)
        scan.add_dicom_info(minimal_info)
        self.assertIn(minimal_info.scan_id, repr(scan))

class TestDicomCollection(unittest.TestCase):
    """Test suite for basic DicomCollection functionality."""

    def setUp(self):
        self.dicom_collection = DicomCollection()
        self.temp_dir = tempfile.mkdtemp()
        self.file_paths, self.test_dir = create_temp_files(TEST_DICOM_FILES, self.temp_dir)

    def tearDown(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)  # Use rmtree to remove directory and all contents

    def test_add_scan(self):
        """Test adding scans with various metadata combinations."""
        # Test with complete metadata
        dicom_info = DicomInfo(
            StudyDescription="Study_01",
            PatientName="Patient_01",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )
        scan = Scan(scan_id=dicom_info.scan_id, dicom_info=dicom_info)
        self.dicom_collection.add_scan(scan)
        self.assertIn(dicom_info.scan_id, self.dicom_collection.scans)

        # Test with minimal metadata
        minimal_info = DicomInfo(SeriesDescription="DWI_b1000")
        scan = Scan(scan_id=minimal_info.scan_id, dicom_info=minimal_info)
        self.dicom_collection.add_scan(scan)
        self.assertIn(minimal_info.scan_id, self.dicom_collection.scans)

    def test_add_invalid_scan(self):
        """Test adding invalid scan types."""
        # Test non-Scan object
        with self.assertRaises(TypeError):
            self.dicom_collection.add_scan("not a scan")

        # Test empty scan ID
        with self.assertRaises(ValueError):
            self.dicom_collection.add_scan(Scan(scan_id=""))

    def test_get_scan(self):
        """Test retrieving scans with various metadata combinations."""
        # Test with complete metadata
        dicom_info = DicomInfo(
            StudyDescription="Study_01",
            PatientName="Patient_01",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )
        scan = Scan(scan_id=dicom_info.scan_id, dicom_info=dicom_info)
        self.dicom_collection.add_scan(scan)
        retrieved_scan = self.dicom_collection.get_scan(dicom_info.scan_id)
        self.assertEqual(retrieved_scan, scan)

        # Test with minimal metadata
        minimal_info = DicomInfo(SeriesDescription="DWI_b1000")
        scan = Scan(scan_id=minimal_info.scan_id, dicom_info=minimal_info)
        self.dicom_collection.add_scan(scan)
        retrieved_scan = self.dicom_collection.get_scan(minimal_info.scan_id)
        self.assertEqual(retrieved_scan, scan)

    def test_populate_from_folder(self):
        """Test populating collection from a folder."""
        file_paths, test_dir = create_temp_files(TEST_DICOM_FILES, self.temp_dir)
        collection = DicomCollection()
        collection.populate_from_folder(test_dir)
        self.assertTrue(collection.scans)  # Verify scans were added

    def test_populate_from_nonexistent_folder(self):
        """Test handling of nonexistent folder."""
        with self.assertRaises(FileNotFoundError):
            self.dicom_collection.populate_from_folder("/nonexistent/path")

    def test_get_unique_scan_names(self):
        """Test retrieving unique scan names with various metadata combinations."""
        # Test with complete metadata
        dicom_info1 = DicomInfo(
            StudyDescription="Study_01",
            PatientName="Patient_01",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )
        dicom_info2 = DicomInfo(
            StudyDescription="Study_02",
            PatientName="Patient_02",
            SeriesDescription="T1_MPRAGE",
            AcquisitionDate="20220101"
        )
        self.dicom_collection.add_scan(Scan(scan_id=dicom_info1.scan_id, dicom_info=dicom_info1))
        self.dicom_collection.add_scan(Scan(scan_id=dicom_info2.scan_id, dicom_info=dicom_info2))
        
        unique_names = self.dicom_collection.get_unique_scan_names()
        self.assertEqual(len(unique_names), 1)
        self.assertEqual(unique_names[0], "T1_MPRAGE")

        # Test with minimal metadata
        minimal_info = DicomInfo()  # No series description
        self.dicom_collection.add_scan(Scan(scan_id=minimal_info.scan_id, dicom_info=minimal_info))
        unique_names = self.dicom_collection.get_unique_scan_names()
        self.assertEqual(len(unique_names), 1)  # Should still only have T1_MPRAGE

    def test_populate_from_empty_folder(self):
        """Test populating collection from a folder with no DICOM files."""
        empty_dir = tempfile.mkdtemp()
        try:
            # Create some non-DICOM files
            with open(os.path.join(empty_dir, 'test.txt'), 'w') as f:
                f.write('not a dicom')
            with open(os.path.join(empty_dir, 'test.dat'), 'w') as f:
                f.write('also not a dicom')
                
            collection = DicomCollection()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                collection.populate_from_folder(empty_dir)
            self.assertEqual(len(collection.scans), 0)  # Verify no scans were added
        finally:
            if os.path.exists(empty_dir):
                shutil.rmtree(empty_dir)

if __name__ == "__main__":
    unittest.main()