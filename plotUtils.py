import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from skimage.util import img_as_float
from skimage.color import gray2rgb
from matplotlib.patches import Polygon
import imageio
import os
from processingUtils import reorient_from_fsl, crop_to_nonzero, apply_crop_bounds
matplotlib.use("Agg")  # Use the Agg backend, which supports tostring_rgb


def construct_displacement_vectors(disp1, disp2, title_str="", axis_labels=None, underlay_image=None, 
                                   fig_width=10, fig_height=10, dpi=200, vmin=None, vmax=None, 
                                   arrow_thickness=0.003, underlay_opacity=0.5):
    """
    Helper function to plot displacement vectors with color scaling by amplitude and save the frame in memory.
    
    Parameters:
    - disp1, disp2: Displacement components along x and y axes.
    - title_str: Title for the plot.
    - axis_labels: Labels for the x and y axes.
    - underlay_image: Background image (grayscale).
    - fig_width, fig_height: Dimensions of the figure in inches.
    - dpi: Dots per inch, for figure resolution.
    - vmin, vmax: Fixed color scale limits for consistent amplitude scaling across frames.
    
    Returns:
    - np.ndarray: Image array representing the plot for use in the video writer.
    """
    downsample_factor = 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Create meshgrid for the original grid size
    X, Y = np.meshgrid(np.arange(disp1.shape[1]), np.arange(disp1.shape[0]))

    # Downsample vectors for visualization
    X_down = X[::downsample_factor, ::downsample_factor]
    Y_down = Y[::downsample_factor, ::downsample_factor]
    disp1_down = disp1[::downsample_factor, ::downsample_factor]
    disp2_down = disp2[::downsample_factor, ::downsample_factor]
    del X, Y

    # Flip Y_down to reverse the direction of the quiver plot in the y-axis
    Y_down = Y_down[::-1, :]

    # Display the underlay image if provided
    if underlay_image is not None:
        underlay_image = img_as_float(underlay_image)
        underlay_image = (underlay_image - underlay_image.min()) / (underlay_image.max() - underlay_image.min())
        
        # Convert grayscale to RGB and apply opacity
        underlay_rgb = gray2rgb(underlay_image) * 0.65
        ax.imshow(underlay_rgb, interpolation='nearest', extent=[0, disp1.shape[1], 0, disp1.shape[0]])
        del underlay_image, underlay_rgb

    # Create quiver plot for displacement vectors
    scale_factor = 120
    Q = ax.quiver(X_down, Y_down, disp1_down * scale_factor, disp2_down * scale_factor, angles='xy', scale_units='xy', scale=1, width=arrow_thickness)
    del X_down, Y_down

    # Normalize based on fixed vmin and vmax
    magnitudes = np.sqrt(disp1_down**2 + disp2_down**2).flatten()
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap="viridis", norm=norm)
    colors = sm.to_rgba(magnitudes)

    # Set alpha to zero for arrows where both disp1 and disp2 are zero
    zero_disp_mask = (disp1_down == 0) & (disp2_down == 0)
    colors[:, 3] = np.where(zero_disp_mask.flatten(), 0, colors[:, 3])

    Q.set_color(colors)
    del magnitudes, colors, Q

    # Remove ticks and numbers on axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set plot details
    ax.set_title(title_str)
    # if axis_labels:
    #     ax.set_xlabel(axis_labels[0])
    #     ax.set_ylabel(axis_labels[1])

    # Add colorbar for magnitude
    # plt.colorbar(sm, ax=ax, label="Magnitude", shrink=0.7, pad=0.05)

    # Remove all padding around the plot
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Render the figure to capture it as a NumPy array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    plt.close(fig)
    return frame


def plot_mri_displacement_vector(mri_displacement, underlay_image=None, mask=None, save_video=True, video_dir='', file_prefix='displacement', padding = 10):
    """
    Plot MRI displacement vectors over time with optional video saving and fixed amplitude scaling.
    
    Parameters:
    - mri_displacement: 5D numpy array of displacement data (x, y, z, displacement components, time).
    - underlay_image: 3D numpy array for background image (optional).
    - mask: 3D binary mask to crop the displacement data (optional).
    - save_video: Whether to save as video (default is True).
    - video_dir: Directory to save the video files.
    - file_prefix: Prefix for saved video filenames.
    """
    print(f"\t\tPlotting {file_prefix}")

    # Reorient images if necessary
    mri_displacement = reorient_from_fsl(mri_displacement)
    if underlay_image is not None:
        underlay_image = reorient_from_fsl(underlay_image)
    if mask is not None:
        mask = reorient_from_fsl(mask)

    # If mask is provided, calculate cropping bounds and crop the displacement data and mask
    if mask is not None:
        if mask.shape[:3] != mri_displacement.shape[:3]:
            raise ValueError('The mask must have the same spatial dimensions as the MRI displacement data.')
        
        # Crop the mask and get cropping bounds
        _, _, crop_bounds = crop_to_nonzero(mask, padding)
        
        # Apply the same cropping bounds to the displacement and underlay
        mri_displacement = apply_crop_bounds(mri_displacement, crop_bounds)
        mask = apply_crop_bounds(mask, crop_bounds)
        if underlay_image is not None:
            underlay_image = apply_crop_bounds(underlay_image, crop_bounds)

        # Scale displacement data based on the mask
        mri_displacement *= mask[..., np.newaxis, np.newaxis]

    # Define the center slices for each dimension
    dims = mri_displacement.shape
    slice_z, slice_y, slice_x = dims[0] // 2, dims[1] // 2, dims[2] // 2

    # Calculate the maximum magnitude for each slice orientation across all time points
    sagittal_magnitude = np.sqrt(
        mri_displacement[:, :, slice_x, 1, :]**2 + mri_displacement[:, :, slice_x, 2, :]**2
    ).max()

    coronal_magnitude = np.sqrt(
        mri_displacement[:, slice_y, :, 0, :]**2 + mri_displacement[:, slice_y, :, 2, :]**2
    ).max()

    axial_magnitude = np.sqrt(
        mri_displacement[slice_z, :, :, 0, :]**2 + mri_displacement[slice_z, :, :, 1, :]**2
    ).max()

    # Get the global maximum magnitude across all orientations
    max_magnitude = max(sagittal_magnitude, coronal_magnitude, axial_magnitude)

    # Set minimum magnitude (assuming zero as the minimum)
    min_magnitude = 0

    # Set up video writers if saving video
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        video_paths = {
            'sagittal': os.path.join(video_dir, f'{file_prefix}_sagittal.mp4'),
            'coronal': os.path.join(video_dir, f'{file_prefix}_coronal.mp4'),
            'axial': os.path.join(video_dir, f'{file_prefix}_axial.mp4')
        }
        video_writers = {
            orientation: imageio.get_writer(path, fps=10, codec='libx264', format='ffmpeg') for orientation, path in video_paths.items()
        }

    # Loop through each time point and slice to construct frames
    for t in range(1,dims[4]):
        # Sagittal (Y-Z plane)
        disp_y = mri_displacement[:, :, slice_x, 1, t]
        disp_z = mri_displacement[:, :, slice_x, 2, t]
        underlay_yz = underlay_image[:, :, slice_x] if underlay_image is not None else None
        frame = construct_displacement_vectors(disp_y, disp_z, f'Sagittal Slice (X axis) Time {t}', ['Y', 'Z'], underlay_yz, vmin=min_magnitude, vmax=max_magnitude)
        if save_video:
            video_writers['sagittal'].append_data(frame)

        # Coronal (X-Z plane)
        disp_x = mri_displacement[:, slice_y, :, 0, t]
        disp_z = mri_displacement[:, slice_y, :, 2, t]
        underlay_xz = underlay_image[:, slice_y, :] if underlay_image is not None else None
        frame = construct_displacement_vectors(disp_x, disp_z, f'Coronal Slice (Y axis) Time {t}', ['X', 'Z'], underlay_xz, vmin=min_magnitude, vmax=max_magnitude)
        if save_video:
            video_writers['coronal'].append_data(frame)

        # Axial (X-Y plane)
        disp_x = mri_displacement[slice_z, :, :, 0, t]
        disp_y = mri_displacement[slice_z, :, :, 1, t]
        underlay_xy = underlay_image[slice_z, :, :] if underlay_image is not None else None
        frame = construct_displacement_vectors(disp_x, disp_y, f'Axial Slice (Z axis) Time {t}', ['X', 'Y'], underlay_xy, vmin=min_magnitude, vmax=max_magnitude)
        if save_video:
            video_writers['axial'].append_data(frame)

    # Close video writers
    if save_video:
        for writer in video_writers.values():
            writer.close()

def calculate_auto_magnification(slice_size, target_display_size):
    """Calculate magnification factor based on slice dimensions and target display size."""
    width = slice_size[1]  # Width in pixels
    return (target_display_size / width) * 100

def plot_mri_slices_time_videos(mri_volume, mask=None, save_video=False, slice_indices=None, video_dir='', file_conv='', outline=False, padding=10, overlay=None, alpha=0.1, fig_width=10, fig_height=10, dpi=200):
    
    print(f"\t\tPlotting {file_conv}")
    
    # Reorient the MRI volume and associated data if necessary
    mri_volume = reorient_from_fsl(mri_volume)
    if mask is not None:
        mask = reorient_from_fsl(mask)
    if overlay is not None:
        overlay = reorient_from_fsl(overlay)

    # Crop the mask and apply cropping bounds if mask is provided
    if mask is not None:
        if mask.shape[:3] != mri_volume.shape[:3]:
            raise ValueError('The mask must have the same spatial dimensions as the MRI displacement data.')
        
        # Crop the mask and get cropping bounds
        _, _, crop_bounds = crop_to_nonzero(mask, padding=padding)
        
        # Apply the cropping bounds to the MRI volume, mask, and overlay (if present)
        mri_volume = apply_crop_bounds(mri_volume, crop_bounds)
        mask = apply_crop_bounds(mask, crop_bounds)
        if overlay is not None:
            overlay = apply_crop_bounds(overlay, crop_bounds)

    dims = mri_volume.shape
    if slice_indices is None:
        slice_x, slice_y, slice_z = dims[0] // 2, dims[1] // 2, dims[2] // 2
    else:
        slice_x, slice_y, slice_z = slice_indices

    # Compute 1% and 99% percentiles across all time points for consistent normalization
    global_min = np.percentile(mri_volume, 0.01)
    global_max = np.percentile(mri_volume, 99.99)

    # Define video file paths
    sagittal_video_path = os.path.join(video_dir, f"{file_conv}_cine_sagittal.mp4")
    axial_video_path = os.path.join(video_dir, f"{file_conv}_cine_axial.mp4")
    coronal_video_path = os.path.join(video_dir, f"{file_conv}_cine_coronal.mp4")

    # Video writer setup
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        video_writers = {
            'sagittal': imageio.get_writer(sagittal_video_path, fps=15, codec='libx264', format='ffmpeg'),
            'axial': imageio.get_writer(axial_video_path, fps=15, codec='libx264', format='ffmpeg'),
            'coronal': imageio.get_writer(coronal_video_path, fps=15, codec='libx264', format='ffmpeg')
        }

    # Plot and save each frame with the time displayed in the title
    for t in range(dims[3]):
        time_title = f"Time {t}"
        frames = {
            'axial': normalize_and_plot_frame(
                mri_volume[:, slice_y, :, t],
                mask[:, slice_y, :] if mask is not None else None,
                overlay[:, slice_y, :] if overlay is not None else None,
                alpha, outline, title=f"Sagittal Slice - {time_title}",
                fig_width=fig_width, fig_height=fig_height, dpi=dpi,
                vmin=global_min, vmax=global_max
            ),
            'sagittal': normalize_and_plot_frame(
                mri_volume[slice_x, :, :, t],
                mask[slice_x, :, :] if mask is not None else None,
                overlay[slice_x, :, :] if overlay is not None else None,
                alpha, outline, title=f"Axial Slice - {time_title}",
                fig_width=fig_width, fig_height=fig_height, dpi=dpi,
                vmin=global_min, vmax=global_max
            ),
            'coronal': normalize_and_plot_frame(
                mri_volume[:, :, slice_z, t],
                mask[:, :, slice_z] if mask is not None else None,
                overlay[:, :, slice_z] if overlay is not None else None,
                alpha, outline, title=f"Coronal Slice - {time_title}",
                fig_width=fig_width, fig_height=fig_height, dpi=dpi,
                vmin=global_min, vmax=global_max
            )
        }
        if save_video:
            for orientation, frame in frames.items():
                video_writers[orientation].append_data(frame)

    # Close video writers
    if save_video:
        for writer in video_writers.values():
            writer.close()

def normalize_and_plot_frame(image_slice, mask_slice=None, overlay_slice=None, alpha=0.1, outline=False, title="", fig_width=10, fig_height=10, dpi=200, vmin=0, vmax=1):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_axis_off()
    norm_slice = (image_slice - vmin) / (vmax - vmin)  # Normalize using global min and max
    
    # Display main image
    ax.imshow(norm_slice, cmap='gray', vmin=0, vmax=1)
    
    # Overlay handling
    if overlay_slice is not None:
        overlay_rgb = apply_overlay(norm_slice, overlay_slice, alpha=alpha)
        ax.imshow(overlay_rgb, alpha=alpha)
    
    # Outline mask
    if outline and mask_slice is not None:
        outline_mask(ax, mask_slice)
    
    plt.title(title)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.canvas.draw()

    # Capture the image from the plot
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame

def normalize_image(img):
    img = img - np.min(img)
    img = img / np.max(img) if np.max(img) != 0 else img
    return img

def apply_overlay(base_img, overlay_img, alpha=0.1, color=[1, 0, 0]):
    base_img = img_as_float(base_img)
    overlay_img = img_as_float(overlay_img)
    overlay_rgb = np.dstack((overlay_img * color[0], overlay_img * color[1], overlay_img * color[2]))
    return (1 - alpha) * np.dstack([base_img]*3) + alpha * overlay_rgb

def outline_mask(ax, mask):
    contours = plt.contour(mask, levels=[0.5], colors='red', linewidths=0.5)
    for collection in contours.collections:
        for path in collection.get_paths():
            verts = path.vertices
            polygon = Polygon(verts, edgecolor='red', fill=None, linewidth=1.5)
            ax.add_patch(polygon)

def analyze_cohesion(displacement_image, mask, save_path="cohesion_metric.png"):
    """
    Calculates and plots a cohesion metric for the vector field at each time point
    by analyzing the eigenvalues of the dispersion tensor (covariance matrix) after
    normalizing each vector to unit length to remove amplitude bias. Saves the plot as a PNG.
    
    Parameters:
    - displacement_image (numpy.ndarray): 5D array of displacement data with dimensions (x, y, z, channels, time).
    - mask (numpy.ndarray): 3D binary mask array with dimensions matching the spatial dimensions of displacement_image.
    - save_path (str): Path to save the plot as a PNG file.
    
    Returns:
    - cohesion_metric (numpy.ndarray): Array of cohesion metrics over time.
    """
    # Apply the mask to the displacement image
    masked_displacement = displacement_image * mask[..., np.newaxis, np.newaxis]
    
    # Get dimensions
    x_dim, y_dim, z_dim, num_channels, num_timepoints = masked_displacement.shape
    
    # Initialize array to store cohesion metrics for each time point
    cohesion_metric = np.zeros(num_timepoints)
    
    # Loop through each time point
    for t in range(num_timepoints):
        # Extract displacement data for the current time point
        data_at_time = masked_displacement[..., t]
        
        # Reshape to combine spatial dimensions into one dimension, keeping channels intact
        reshaped_data = data_at_time.reshape(-1, num_channels)
        
        # Remove rows with all zeros (outside the brain region)
        reshaped_data_nonzero = reshaped_data[np.any(reshaped_data, axis=1)]
        
        # Normalize each vector to unit length to remove amplitude bias
        vector_magnitudes = np.linalg.norm(reshaped_data_nonzero, axis=1, keepdims=True)
        normalized_vectors = reshaped_data_nonzero / vector_magnitudes
        
        # Compute the covariance matrix (dispersion tensor) for the normalized vectors
        covariance_matrix = np.cov(normalized_vectors, rowvar=False)
        
        # Calculate the eigenvalues of the covariance matrix
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        
        # Calculate the dispersion metric as the sum of eigenvalues
        dispersion_metric = np.sum(eigenvalues)
        
        # Calculate cohesion as the inverse of dispersion metric
        cohesion_metric[t] = 1 / dispersion_metric if dispersion_metric != 0 else np.inf
    
    # Plot the cohesion metric over time and save as a PNG
    plt.figure()
    plt.plot(range(1, num_timepoints + 1), cohesion_metric, marker='o')
    plt.title('Directional Cohesion Metric Over Time (Amplitude-Normalized)')
    plt.xlabel('Time Point')
    plt.ylabel('Cohesion Metric (Higher is More Coherent)')
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()  # Close the plot to free up memory