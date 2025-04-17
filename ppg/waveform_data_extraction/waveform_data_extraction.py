##### I have done this in colab just for me to try, will have to adjust to aws structure #######


import os
import json
import pandas as pd
import numpy as np
import requests
import concurrent.futures
from tqdm import tqdm

# Google Colab Integration (if using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    SAVE_DIR = "/content/drive/My Drive/mimic3_ppg_data"
except ImportError:
    SAVE_DIR = "mimic3_ppg_data"

os.makedirs(SAVE_DIR, exist_ok=True)

# Base URL for MIMIC-III waveform database
BASE_URL = "https://physionet.org/files/mimic3wdb/1.0/"

def download_file(url, save_path):
    """Download a file from a given URL."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {save_path}")
            return True
        else:
            print(f"❌ Failed: {url} (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Exception for {url}: {e}")
        return False

def download_record_with_segments(directory, record_id, max_segments=5):
    """
    Download record files with numbered segments.

    Args:
        directory: The directory containing the record (e.g., "30")
        record_id: The record ID (e.g., "3000051")
        max_segments: Maximum number of segments to try downloading

    Returns:
        Dictionary with download results
    """
    record_dir = os.path.join(SAVE_DIR, f"{directory}_{record_id}")
    os.makedirs(record_dir, exist_ok=True)

    # Results to track downloads
    results = {
        "directory": directory,
        "record_id": record_id,
        "success": False,
        "files_downloaded": [],
        "segments_found": 0,
        "has_pleth": False
    }

    # First, download the header and layout files
    header_files = [
        f"{record_id}.hea",
        f"{record_id}_layout.hea"
    ]

    for file_name in header_files:
        file_url = f"{BASE_URL}{directory}/{record_id}/{file_name}"
        save_path = os.path.join(record_dir, file_name)
        if download_file(file_url, save_path):
            results["files_downloaded"].append(file_name)

    # Check if the layout file contains PLETH signals
    layout_path = os.path.join(record_dir, f"{record_id}_layout.hea")
    if os.path.exists(layout_path):
        try:
            with open(layout_path, 'r') as f:
                layout_content = f.read()
            results["has_pleth"] = "PLETH" in layout_content
        except Exception as e:
            print(f"Error reading layout file: {e}")

    # Download numbered segments (0001, 0002, etc.)
    for segment_num in range(1, max_segments + 1):
        segment_id = f"{record_id}_{segment_num:04d}"
        segment_files = [
            f"{segment_id}.dat",
            f"{segment_id}.hea"
        ]

        segment_found = False
        for file_name in segment_files:
            file_url = f"{BASE_URL}{directory}/{record_id}/{file_name}"
            save_path = os.path.join(record_dir, file_name)
            if download_file(file_url, save_path):
                results["files_downloaded"].append(file_name)
                segment_found = True

        if segment_found:
            results["segments_found"] += 1

    results["success"] = results["segments_found"] > 0 and results["has_pleth"]
    return results

def main():
    """Main function to download PPG waveform data"""
    print("\nMIMIC-III PPG Waveform Downloader (Working Version)")
    print("=" * 70)

    # Check if cache file exists
    cache_file = os.path.join(SAVE_DIR, "cache", "ppg_cache_30_341.json")
    if os.path.exists(cache_file):
        # Load the cache file
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        ppg_segments = cache_data["ppg_segments"]
        print(f"Loaded {len(ppg_segments)} PPG segments from cache.")

        # Ask how many segments to download
        try:
            max_segments = int(input(f"How many PPG segments to download (max {len(ppg_segments)})? "))
            max_segments = min(max(1, max_segments), len(ppg_segments))
        except ValueError:
            max_segments = min(5, len(ppg_segments))
            print(f"Invalid input. Using default value: {max_segments}")

        # Select segments to process
        segments_to_download = ppg_segments[:max_segments]
    else:
        print("Cache file not found. Please enter records manually.")
        directory = input("Enter directory (default: 30): ") or "30"

        record_ids_text = input("Enter record IDs separated by commas (default: 3000051): ") or "3000051"
        record_ids = [rid.strip() for rid in record_ids_text.split(",")]

        segments_to_download = [{"record_path": f"{directory}/{record_id}", "segment": record_id} for record_id in record_ids]
        max_segments = len(segments_to_download)

    print(f"Will download {len(segments_to_download)} PPG segments.")

    # Process each segment
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_segment = {}

        for segment in segments_to_download:
            # Parse the record path to get directory and record ID
            parts = segment["record_path"].split("/")
            directory = parts[0]
            record_id = segment["segment"]

            future = executor.submit(download_record_with_segments, directory, record_id)
            future_to_segment[future] = segment

        for future in tqdm(concurrent.futures.as_completed(future_to_segment),
                          total=len(future_to_segment),
                          desc="Downloading waveform data"):
            segment = future_to_segment[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing segment {segment['segment']}: {e}")

    # Generate summary statistics
    successful = [r for r in results if r["success"]]
    print(f"\nSuccessfully downloaded {len(successful)} out of {len(segments_to_download)} segments")

    if successful:
        total_segments = sum(r["segments_found"] for r in successful)
        print(f"Total segments found: {total_segments}")
        print(f"Average segments per record: {total_segments / len(successful):.1f}")

        # Count the types of files downloaded
        file_types = {}
        for result in successful:
            for file in result["files_downloaded"]:
                file_type = file.split(".")[-1]
                file_types[file_type] = file_types.get(file_type, 0) + 1

        print("\nFiles downloaded by type:")
        for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {file_type}: {count} files")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(SAVE_DIR, "waveform_download_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved download results to: {results_csv}")

    # Provide guidance on how to use the downloaded data with WFDB
    if successful:
        sample_record = successful[0]
        print("\nTo analyze these waveforms with WFDB in Python:")
        print("```python")
        print("import wfdb")
        print("import matplotlib.pyplot as plt")
        print()
        print("# Install WFDB if needed: pip install wfdb")
        print()
        print(f"# Load a record (replace with your path)")
        directory = sample_record['directory']
        record_id = sample_record['record_id']
        record_path = os.path.join(SAVE_DIR, f"{directory}_{record_id}", f"{record_id}_0001")
        print(f"record_path = '{record_path}'")
        print("record = wfdb.rdrecord(record_path)")
        print()
        print("# Find PLETH signals")
        print("pleth_indices = [i for i, name in enumerate(record.sig_name) if 'PLETH' in name]")
        print("if pleth_indices:")
        print("    # Extract PLETH data")
        print("    pleth_data = record.p_signal[:, pleth_indices[0]]")
        print("    ")
        print("    # Plot PLETH signal")
        print("    plt.figure(figsize=(15, 5))")
        print("    plt.plot(pleth_data)")
        print("    plt.title('PPG Waveform')")
        print("    plt.xlabel('Samples')")
        print("    plt.ylabel('Amplitude')")
        print("    plt.show()")
        print("```")

if __name__ == "__main__":
    main()

### Inspect what was downloaded ####

# Google Colab Integration (if using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    SAVE_DIR = "/content/drive/My Drive/mimic3_ppg_data"
except ImportError:
    SAVE_DIR = "mimic3_ppg_data"

import os
os.makedirs(SAVE_DIR, exist_ok=True)

import wfdb

# Adjust these variables as needed.
# Here, we assume a folder "30_3000051" exists in SAVE_DIR with record "3000051"
record_dir = os.path.join(SAVE_DIR, "30_3000051")
record_name = "3000051"
record_path = os.path.join(record_dir, record_name)

try:
    record = wfdb.rdrecord(record_path)
    print("Record Header Info:")
    print(record.__dict__)

    print("\nFirst 10 samples of all channels:")
    print(record.p_signal[:10])

    # If you're interested in the PPG channel (usually the last channel, index 5)
    print("\nFirst 10 samples of the PPG channel (PLETH):")
    print(record.p_signal[:10, 5])

    # Check the tail of the PPG channel to see if valid data exists later on
    print("\nLast 10 samples of the PPG channel (PLETH):")
    print(record.p_signal[-10:, 5])

except Exception as e:
    print(f"Error reading record {record_name}: {e}")

import numpy as np
import pandas as pd

# Extract the PPG channel (assuming it's the last one)
ppg = record.p_signal[:, 5]

# Find the first index where data isn't NaN
first_valid = np.argmax(~np.isnan(ppg))
print("First valid sample index:", first_valid)

# Trim the signal from the first valid index onward
trimmed_ppg = ppg[first_valid:]

# Optionally, if there are sporadic NaNs later, interpolate them:
clean_ppg = pd.Series(trimmed_ppg).interpolate(limit_direction='both').to_numpy()

print("Cleaned PPG signal (first 10 samples):")
print(clean_ppg[:10])


import os
import wfdb
import matplotlib.pyplot as plt

# Google Colab Integration (if using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    SAVE_DIR = "/content/drive/My Drive/mimic3_ppg_data"
except ImportError:
    SAVE_DIR = "mimic3_ppg_data"

# Get all record folders that start with "30_"
record_folders = [f for f in os.listdir(SAVE_DIR) if f.startswith("30_") and os.path.isdir(os.path.join(SAVE_DIR, f))]
print("Found records:", record_folders)

for folder in record_folders:
    folder_path = os.path.join(SAVE_DIR, folder)
    print("\nRecord:", folder)

    # Get segment files (ignore layout files)
    segment_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".hea") and "layout" not in f])

    if not segment_files:
        print(" No segment files found.")
        continue

    for seg_file in segment_files:
        seg_name = seg_file[:-4]  # remove ".hea"
        seg_path = os.path.join(folder_path, seg_name)
        try:
            seg_record = wfdb.rdrecord(seg_path)
            if "PLETH" in seg_record.sig_name:
                pleth_index = seg_record.sig_name.index("PLETH")
                pleth_channel = seg_record.p_signal[:, pleth_index]
                print(f" Segment {seg_name} - First 10 samples of PLETH channel:")
                print(pleth_channel[:10])

                # Plot for inspection
                plt.figure(figsize=(10, 3))
                plt.plot(pleth_channel)
                plt.title(f"{folder} - {seg_name} - PLETH Channel")
                plt.xlabel("Sample")
                plt.ylabel("Amplitude")
                plt.show()
            else:
                print(f" Segment {seg_name}: PLETH channel not found.")
        except Exception as e:
            print(f" Error loading segment {seg_name} in record {folder}: {e}")
