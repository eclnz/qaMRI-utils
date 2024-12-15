# import nibabel as nib
# import os
# import numpy as np
# from typing import Optional
# from plotUtils2 import GeneralMRIPlotter

# def load_nifti_data(file_path: str) -> np.ndarray:
#     """Loads a NIfTI file and returns the data as a NumPy array."""
#     nifti_img = nib.load(file_path)
#     return nifti_img.get_fdata(dtype=np.float32)  # type: ignore

# def test_mri_plotting(file_path_3d: str, file_path_4d: str, file_path_5d: Optional[str] = None, underlay_path: Optional[str] = None):
#     output_dir = "mri_plots"

#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Load the optional underlay image if provided
#     underlay_image = load_nifti_data(underlay_path) if underlay_path else None

#     mask_image = load_nifti_data(mask_path) if mask_path else None

#     # Test with 3D MRI data
#     print("Testing with 3D MRI data...")
#     mri_data_3d = load_nifti_data(file_path_3d)
#     plotter_3d = GeneralMRIPlotter(mri_data_3d, output_dir=output_dir, mask=mask_image)
#     plotter_3d.plot()
#     print("3D MRI plotting complete. Output saved to:", output_dir)

#     # Test with 4D MRI data
#     print("Testing with 4D MRI data...")
#     mri_data_4d = load_nifti_data(file_path_4d)
#     plotter_4d = GeneralMRIPlotter(mri_data_4d, output_dir=output_dir, mask=mask_image, crop=True)
#     plotter_4d.plot()
#     print("4D MRI plotting complete. Output saved to:", output_dir)

#     # Test with 5D MRI data, if available
#     if file_path_5d:
#         print("Testing with 5D MRI data...")
#         mri_data_5d = load_nifti_data(file_path_5d)
#         plotter_5d = GeneralMRIPlotter(mri_data_5d, output_dir=output_dir, underlay_image=underlay_image, mask=mask_image, crop=True, mask_underlay=True)
#         plotter_5d.plot()
#         print("5D MRI plotting complete. Output saved to:", output_dir)
#     else:
#         print("5D MRI file not provided; skipping 5D test.")

# # Example NIfTI file paths for testing
# # Replace these paths with your actual file paths
# file_path_5d = "/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4/derivatives/aMRI/sub-control1/ses-2021B/sub-control1_ses-2021B_desc-motion_map.nii.gz"
# file_path_4d = "/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4/derivatives/aMRI/sub-control1/ses-2021B/sub-control1_ses-2021B_desc-aMRI_amplified.nii.gz"
# file_path_3d = "/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4/derivatives/aMRI/sub-control1/ses-2021B/sub-control1_ses-2021B_desc-aMRI_reorient_first.nii.gz"
# underlay_path = file_path_4d
# mask_path = "/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4/derivatives/aMRI/sub-control1/ses-2021B/sub-control1_ses-2021B_desc-aMRI_resizePad_mask.nii.gz"


# # Run the test
# test_mri_plotting(file_path_3d, file_path_4d, file_path_5d, underlay_path)