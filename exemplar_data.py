import urllib.request
import shutil
import tempfile
import os

def download_test_dicom():
    """Download a test DICOM file from pydicom's test data."""
    dicom_url = "https://github.com/dangom/sample-dicom/blob/master/MR000000.dcm"
    temp_dir = tempfile.mkdtemp()
    local_file = os.path.join(temp_dir, "test.dcm")
    
    try:
        if not dicom_url.startswith('https://'):
            raise ValueError("URL must use HTTPS")
        request = urllib.request.Request(dicom_url, headers={'User-Agent': 'Mozilla/5.0'}, method='GET')
        with urllib.request.urlopen(request) as response:
            with open(local_file, 'wb') as f:
                shutil.copyfileobj(response, f)
        return local_file, temp_dir
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to download test DICOM file: {e}") from e