"""Remove house light contamination from two-photon microscopy data using unsupervised machine learning.

This tool automatically identifies saturated pixels using K-means clustering and replaces
them with background pixels from low-fluorescence frames, preserving image structure.

See README.md for detailed usage and examples.
""" 

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import h5py
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from typing import Optional
import stat

def make_file_readonly(file_path):
    """Make a file read-only. This is optional"""
    os.chmod(file_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"Made {file_path} read-only")

def load_tiff(input_tiff):
    """Load a multiframe TIFF file as a 3D numpy array.

    Args:
        input_tiff (str): Path to the original TIFF file with house light contamination.

    Returns:
        np.ndarray: 3D numpy array with shape (num_frames, height, width).
    """
    with tifffile.TiffFile(input_tiff) as tiff:
        img_data = np.stack([page.asarray() for page in tqdm(tiff.pages, desc="Loading TIFF pages")])

    print(f"Loaded TIFF file with shape: {img_data.shape}")

    return img_data

def plot_kmeans_scatter(sampled_pixel_values: np.ndarray, labels: np.ndarray, houselight_label: int, threshold: float, kmeans_output_folder: str) -> None:
    """Subsample the data and plot as a scatter plot.
    
    Args:
        sampled_pixel_values (np.ndarray): Array of pixel values.
        labels (np.ndarray): Cluster labels for the pixels.
        houselight_label (int): Label corresponding to house light.
        threshold (float): Threshold value distinguishing house light.
        kmeans_output_folder (str): Directory to save the plot.
    """
    
    # Make 1D for plotting
    sampled_pixel_values = sampled_pixel_values.ravel()

    print("Sampled Pixel Values Shape After Raveling:", sampled_pixel_values.shape)
    print("Labels Shape:", labels.shape)
    
    # Subsample for faster plotting
    num_points_to_plot = 1000
    if len(sampled_pixel_values) > num_points_to_plot:
        indices = np.random.choice(len(sampled_pixel_values), num_points_to_plot, replace=False)
        sampled_pixel_values = sampled_pixel_values[indices]
        labels = labels[indices]
    
    # Ensure labels and pixel values have the same length
    assert len(sampled_pixel_values) == len(labels), "Mismatch between pixel values and labels!"

    # Plot the subsampled data
    output_path = os.path.join(kmeans_output_folder, 'kmeans_clustering_scatterplot.png')
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, sampled_pixel_values, c='black', s=1, alpha=0.7)    
    plt.axhline(y=threshold, color='darkgrey', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title('K-Means Clustering of Pixel Data (Subsampled)')
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Value')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


    print(f"K-means clustering plot saved to {kmeans_output_folder}")


def find_house_light_threshold(img_data: np.ndarray, kmeans_output_folder: str, num_sample_frames: int = 10000) -> float:
    """Identify the house light threshold using K-means clustering.

    Saves scatter plot for kmeans and a text file with the details of kmeans clustering.

    Args:
        img_data (np.ndarray): 3D numpy array of image data.
        kmeans_output_folder (str): Directory to save K-means results.
        num_sample_frames (int, optional): Number of frames to sample for K-means. Defaults to 10000.

    Returns:
        float: The calculated threshold value.
    """
    # Get the total number of frames
    total_frames = img_data.shape[0]

    # Randomly select a subset of frames to make kmeans faster
    if total_frames > num_sample_frames:
        sampled_frames = np.random.choice(total_frames, num_sample_frames, replace=False)
    else:
        sampled_frames = np.arange(total_frames)

    # Flatten the selected frames to 1D pixel values
    sampled_pixel_values = img_data[sampled_frames].reshape(-1, 1)

    # Apply K-means with 2 clusters
    print('Applying K-means clustering...')
    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=10000, n_init=1, random_state=0).fit(sampled_pixel_values)

    print('K-means clustering complete')

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    # Identify house light cluster
    cluster_0 = sampled_pixel_values[labels == 0]
    cluster_1 = sampled_pixel_values[labels == 1]
    if len(cluster_0) < len(cluster_1):
        houselight_center, trial_center = centers[0], centers[1]
        houselight_label = 0
        trial_label = 1
    else:
        houselight_center, trial_center = centers[1], centers[0]
        houselight_label = 1
        trial_label = 0

    # Calculate threshold
    threshold = (houselight_center + trial_center) / 2
    print('House light threshold:', threshold)

    # Save results to text file
    output_file = os.path.join(kmeans_output_folder, 'kmeans_results.txt')
    with open(output_file, "w") as f:
        f.write(f"Houselight Cluster Center: {houselight_center}\n")
        f.write(f"Trial Cluster Center: {trial_center}\n")
        f.write(f"Threshold: {threshold}\n")
    print(f"K-means clustering results saved to {output_file}")
    
    return threshold

def filter_valid_frames(img_data: np.ndarray, threshold: float, output_folder: str) -> np.ndarray:
    """Identify and filter frames with all pixels below the threshold.
    
    Args:
        img_data (np.ndarray): 3D numpy array with shape (num_frames, height, width).
        threshold (float): Value above which pixels are considered house light.
        output_folder (str): Directory to save valid frame indices.

    Returns:
        np.ndarray: 3D numpy array excluding frames that have houselight.
    """
    print('filtering frames')
    valid_frame_mask = np.all(img_data <= threshold, axis=(1, 2))
    valid_frames = img_data[valid_frame_mask]
    if len(valid_frames) == 0:
        raise ValueError("No frames meet the criteria of having all pixels below the threshold.")

    # Extract indices of valid frames
    valid_frame_indices = np.where(valid_frame_mask)[0]

    np.save(os.path.join(output_folder, "valid_frames_indices.npy"), valid_frame_indices)
    # Save to .csv file
    csv_path = os.path.join(output_folder, "valid_frames_indices.csv")
    np.savetxt(csv_path, valid_frame_indices, delimiter=",", fmt="%d", header="Valid_Frame_Indices", comments="")
    
    print(f"Valid frames saved to: {csv_path}")

    return valid_frames

def save_valid_frame_indices(valid_frame_indices: np.ndarray, output_folder: str, filename: str) -> None:
    """Save valid frame indices to both a CSV and NPY file.

    Args:
        valid_frame_indices (np.ndarray): Indices of valid frames.
        output_folder (str): Path to save the files.
        filename (str): Base filename for the saved files (without extension).
    """
    # Flatten the valid frame indices for saving
    valid_frame_indices = valid_frame_indices.flatten()

    # Save to .npy file
    npy_path = os.path.join(output_folder, f"{filename}_valid_indices.npy")
    np.save(npy_path, valid_frame_indices)
    print(f"Valid frame indices saved to: {npy_path}")

    # Save to .csv file
    csv_path = os.path.join(output_folder, f"{filename}_valid_indices.csv")
    np.savetxt(csv_path, valid_frame_indices, delimiter=",", fmt="%d", header="Valid_Frame_Indices", comments="")
    print(f"Valid frame indices saved to: {csv_path}")

def calculate_fluorescence(valid_frames: np.ndarray, method: str = 'sum') -> np.ndarray:
    """Calculate fluorescence values for valid frames using the specified method.
    
    Args:
        valid_frames (np.ndarray): 3D numpy array excluding frames with houselight.
        method (str, optional): Method to calculate fluorescence ('sum' or 'mean'). Defaults to 'sum'.

    Returns:
        np.ndarray: 1D numpy array of fluorescence values for each valid frame.
    """
    if method == 'mean':
        print('Calculating mean fluorescence for valid frames...')
        return np.mean(valid_frames, axis=(1, 2))
    elif method == 'sum':
        print('Calculating sum fluorescence for valid frames...')
        return np.sum(valid_frames, axis=(1, 2))
    else:
        raise ValueError("Invalid method. Choose 'mean' or 'sum'.")

def plot_cdf(data: np.ndarray, output_path: str, method: str, p: float) -> None:
    """Plot and save the Cumulative Density Function (CDF) of the given data.
    
    Args:
        data (np.ndarray): Data to plot.
        output_path (str): Path to save the plot.
        method (str): Method used for fluorescence calculation (for labeling).
        p (float): Percentile threshold value.
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_data, cdf, label="CDF", linewidth=2)

    # Add horizontal dashed line at y=p in dark grey
    plt.axhline(y=p*.01, color='darkgrey', linestyle='--', linewidth=1, alpha=0.9)

    # Add text just above the dashed line
    plt.text(sorted_data[int(len(sorted_data) * 0.99)], p*.01 + 0.02, "Low Frame Threshold", color='darkgrey', ha='right', fontsize=10)
    plt.xlabel(f"{method} Fluorescence")
    plt.ylabel("Cumulative Density")
    plt.title(f"CDF of {method} Fluorescence")
    #plt.grid(True, linestyle="--", alpha=0.7)
    #plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"CDF plot saved to {output_path}")

def select_low_fluorescence_frames(valid_frames: np.ndarray, fluorescence_values: np.ndarray, p: float) -> np.ndarray:
    """Select p% of valid frames with the lowest fluorescence values.
    
    Args:
        valid_frames (np.ndarray): 3D array of valid frames.
        fluorescence_values (np.ndarray): Corresponding fluorescence values.
        p (float): Percentage of lowest frames to select.

    Returns:
        np.ndarray: The selected low fluorescence frames.
    """

    print(f"Selecting {p}% of lowest fluorescence frames")
    num_low_frames = max(1, round(len(valid_frames) * p // 100))
    # if num_flow_frames is not an integer, round 
    lowest_frame_indices = np.argsort(fluorescence_values)[:num_low_frames]
    return valid_frames[lowest_frame_indices]

def save_tiff(data: np.ndarray, path: str) -> None:
    """Save a 3D numpy array as a TIFF file.
    
    Args:
        data (np.ndarray): Image data to save.
        path (str): Destination file path.
    """
    tifffile.imwrite(path, data)
    print(f"Saved TIFF file to: {path}")

def process_frames(img_data: np.ndarray, low_frames: np.ndarray, threshold: float) -> np.ndarray:
    """Replace pixels above the threshold with those from random low-mean frames.
    
    Args:
        img_data (np.ndarray): Original image data.
        low_frames (np.ndarray): Pool of low fluorescence frames for replacement.
        threshold (float): Threshold value for identifying house light.

    Returns:
        np.ndarray: Processed image data with house light replaced.
    """
    processed_frames = np.empty_like(img_data)
    print('Replacing house light pixels...')
    for i, frame in enumerate(img_data):
        mask = frame > threshold
        random_frame = low_frames[np.random.randint(0, len(low_frames))]
        frame[mask] = random_frame[mask]
        processed_frames[i] = frame
    return processed_frames

def shift_frames(processed_frames: np.ndarray) -> np.ndarray:
    """Offset the histogram so that it starts roughly around 0.
    (The offset can be an odd value if the acquisiton began with contamination and 
    the acquisition software scales pixel values) 
    
    Args:
        processed_frames (np.ndarray): Image data to shift.

    Returns:
        np.ndarray: Shifted image data.
    """
    # Calculate the minimum value in the array
    quantile = 0.05
    min_value = np.quantile(processed_frames, quantile)
    shifted_frames = processed_frames + abs(min_value)

    return shifted_frames

def save_hdf5(data: np.ndarray, path: str) -> None:
    """Save a 3D numpy array to an HDF5 file.
    
    Args:
        data (np.ndarray): Data to save.
        path (str): Destination file path.
    """
    with h5py.File(path, "w") as hdf5_file:
        hdf5_file.create_dataset("image_data", data=data)
    print(f"Saved HDF5 file to: {path}")


def plot_histogram(data: np.ndarray, output_path: str) -> None:
    """Plot and save a histogram of the given data.
    
    Args:
        data (np.ndarray): Data to plot.
        output_path (str): Directory to save the histogram.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=50, color='blue', alpha=0.7)
    plt.title("Histogram of Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_path, "HistogramProcessedTiff.png"))
    plt.close()
    print(f"Histogram saved to {output_path}")

def process_tiff(input_tiff: str, output_folder: str, filename: str, threshold: Optional[float] = None, method: str = 'sum', p: float = 5) -> None:
    """Process a multiframe TIFF file by replacing pixels above a threshold.
    
    Args:
        input_tiff (str): Path to the input TIFF file.
        output_folder (str): Directory to save output files.
        filename (str): Base filename for outputs.
        threshold (Optional[float]): House light threshold. If None, calculated via K-means.
        method (str, optional): Fluorescence calculation method. Defaults to 'sum'.
        p (float, optional): Percentage of low frames to use. Defaults to 5.
    """
    
    print(f'Processing {filename}')

    # load the tiff
    img_data = load_tiff(input_tiff)

    # use Kmeans to find the threshold for the houselight
    kmeans_output_folder = f"{output_folder}/kmeans_output"
    os.makedirs(kmeans_output_folder, exist_ok=True)
    if threshold is None: 
        threshold = find_house_light_threshold(img_data, kmeans_output_folder, num_sample_frames=5000)

    # Use output_folder for all generated files
    valid_frames = filter_valid_frames(img_data, threshold, output_folder) # frames with no houselight 
    fluorescence_values = calculate_fluorescence(valid_frames, method)

    # Plot a CDF of the fluorescence for non houselight frames values
    cdf_output_path = os.path.join(output_folder, f"{filename}_{method}_fluorescence_cdf.png")
    plot_cdf(fluorescence_values, cdf_output_path, method, p)

    # find the low frames that will be used to replace houselight
    low_frames = select_low_fluorescence_frames(valid_frames, fluorescence_values, p)
    
    # Create folder for lowest frames
    low_frames_subfolder = os.path.join(output_folder, f"lowest_{p}_frames")
    os.makedirs(low_frames_subfolder, exist_ok=True)

    # save the low frames as a tiff if needed
    low_frames_path = os.path.join(low_frames_subfolder, f"{filename}_lowest{p}Frames.tiff")
    save_tiff(low_frames, low_frames_path)

    # Main processing of the tiff. Apply a mask using the threshold and replace the houselight
    processed_frames = process_frames(img_data, low_frames, threshold)

    # shift the frames if the offset is wrong 
    processed_frames = shift_frames(processed_frames)

    # Save HDF5
    h5_filepath = os.path.join(output_folder, f"{filename}_HLprocessed.h5")
    save_hdf5(processed_frames, h5_filepath)
    print('H5 file save at: ', h5_filepath)

    ## save final tiff 
    tiff_output_path = os.path.join(output_folder, f"{filename}_HLprocessed.tif")
    save_tiff(processed_frames, tiff_output_path)
    print('Processed tiff file saved at: ', tiff_output_path)

def main():
    parser = argparse.ArgumentParser(description="Replace house light contamination in microscopy TIFF files.")
    
    parser.add_argument('--input_tiff', type=str, required=True, help='Path to the input TIFF file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Directory to save the output files.')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for house light (optional, calculated via K-means if not provided).')
    parser.add_argument('--percent_low_frames', type=float, default=2.5, help='Percentage of low fluorescence frames to use for replacement (default: 2.5).')
    parser.add_argument('--method', type=str, choices=['sum', 'mean'], default='sum', help='Method to calculate fluorescence (default: sum).')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)

    filename = os.path.splitext(os.path.basename(args.input_tiff))[0]

    process_tiff(
        input_tiff=args.input_tiff, 
        output_folder=args.output_folder, 
        filename=filename, 
        threshold=args.threshold, 
        method=args.method, 
        p=args.percent_low_frames
    )

if __name__ == "__main__":
    main()
