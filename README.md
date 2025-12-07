# HandleHouseLight

## Overview

**replace_houselight.py** uses unsupervised machine learning to automatically identify and repair "house light" contamination in two-photon microscopy data.

Light contamination can result in saturated pixels that are outside the regular imaging distribution. These artifacts can cause automated segmentation tools to crash or produce erroneous results. Since most sensors (such as GCaMP6f) have some baseline fluorescence, simply masking these pixels with black or random noise can fundamentally alter the image structure which also cause erroneous results with automated segmentation tools.

This tool solves this problem by:
1. Automatically identifying contaminated pixels using unsupervised learning (K-means clustering).
2. Creating a "bank" of replacement pixels from the darkest (lowest fluorescence) valid frames in the same recording.
3. Replacing contaminated pixels with valid background pixels from the same spatial location, preserving the image's structural noise properties.

## Requirements

The script requires Python 3 and the following libraries:

- `numpy`
- `tifffile`
- `matplotlib`
- `h5py`
- `scikit-learn`
- `tqdm`

You can install them via pip:

```bash
pip install numpy tifffile matplotlib h5py scikit-learn tqdm
```

## Usage

Run the script from the command line by providing the input TIFF file and the desired output folder.

### Basic Example

```bash
python replace_houselight.py --input_tiff /path/to/imaging/data.tif --output_folder /path/to/output/
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_tiff` | `str` | **Required** | Path to the input TIFF file containing the recording. |
| `--output_folder` | `str` | **Required** | Directory where all output files and plots will be saved. |
| `--threshold` | `float` | `None` | Manually set the pixel value threshold for house light. If not provided, it is calculated automatically using K-means clustering. |
| `--percent_low_frames` | `float` | `2.5` | The percentage of "quiet" (low fluorescence) frames to use as a source for replacement pixels. |
| `--method` | `str` | `'sum'` | Method to calculate fluorescence for frame selection. Options: `'sum'` or `'mean'`. |

## How It Works

1. **Detection**: The script samples pixel values from the video and uses **K-means clustering** (k=2) to separate "signal" pixels from "house light" (saturated) pixels. It calculates a threshold based on these clusters.
2. **Frame Selection**: It identifies frames that are completely free of house light ("valid frames"). From these, it selects the bottom $p\%$ (default 2.5%) based on total fluorescence to serve as a background noise bank.
3. **Replacement**: For every pixel identified as "house light" in a contaminated frame, the script replaces it with a pixel value from the exact same spatial coordinate taken randomly from one of the background bank frames.
4. **Output**: The cleaned data is saved as both an HDF5 file (`.h5`) for efficient processing and a TIFF file (`.tif`).

## Outputs

The script generates several files in the output directory:

- **`*_HLprocessed.h5`**: The final cleaned video in HDF5 format.
- **`*_HLprocessed.tif`**: The final cleaned video in TIFF format.
- **`kmeans_output/`**: Contains a scatter plot and text file detailing the threshold calculation.
- **`*_valid_indices.csv/npy`**: Indices of frames that were free of contamination.
- **`*_fluorescence_cdf.png`**: A plot showing the distribution of fluorescence values and the cutoff for "low frames."
- **`lowest_*_frames/`**: A subfolder containing the bank of frames used for replacement.

