import os
import pandas as pd
import numpy as np

#Settings

#fs = 30Hz
#128 samples = 4.266667 seconds

n = 512        # window size (samples)
overlap = int(n*(3/4))  # amount of overlap between windows (samples)
stride = n - overlap

input_folder = "Z:\\Eversource Data Files MinMax Normalized"
output_folder_normal = \
    f"Z:\\Chunked Data\\0.75 overlap\\Chunked Example Data MinMax Normal {n} Sample Windows {overlap} Overlap with Backshift 2"
output_folder_oscillation = \
    f"Z:\\Chunked Data\\0.75 overlap\\Chunked Example Data MinMax Oscillation {n} Sample Windows {overlap} Overlap with Backshift 2"



# State ranges (1-indexed sample numbers)
state1_ranges = [(1, 8400), (40380, 54000)] #data ranges of normal state
state2_ranges = [(8401, 40380)] #data ranges of oscillation state

all_state_ranges = {
    1: state1_ranges,
    2: state2_ranges
}

os.makedirs(output_folder_normal, exist_ok=False)
os.makedirs(output_folder_oscillation, exist_ok=False)


def generate_windows_for_region(series, r_start, r_end, state, file, col, output_folder, n, stride):
    """
    Generate fixed-size windows from r_start to r_end (1-indexed),
    using given stride and applying backward-overlap for the final window.
    """
    region_len = r_end - r_start + 1

    # Zero-index view into region
    region = series[r_start - 1 : r_end]

    windows = []
    ws = 0  # region-local start index (0-indexed)

    # -----------------------------
    # Forward windows using stride
    # -----------------------------
    while ws + n <= region_len:
        windows.append((ws, ws + n))
        ws += stride

    # -----------------------------
    # Final window (ensure full coverage)
    # -----------------------------
    last_end_needed = region_len
    last_start_needed = last_end_needed - n

    if last_start_needed < 0:
        last_start_needed = 0  # clip to region start

    # Only add final window if it’s not already included
    if not windows or windows[-1][1] != region_len:
        windows.append((last_start_needed, last_start_needed + n))

    # -----------------------------
    # Save all windows
    # -----------------------------
    for (ws, we) in windows:
        window = region.iloc[ws:we]

        # Convert back to global sample numbers (1-indexed)
        global_start = r_start + ws
        global_end = r_start + we - 1

        if state == 1:
            out_filename = (
                f"{os.path.splitext(file)[0]}_{col}_normal_"
                f"{global_start}-{global_end}.npy"
            )
            window.to_numpy()
            np.save(os.path.join(output_folder_normal, out_filename), window)

        if state == 2:
            out_filename = (
                f"{os.path.splitext(file)[0]}_{col}_oscillation_"
                f"{global_start}-{global_end}.npy"
            )
            window.to_numpy()
            np.save(os.path.join(output_folder_oscillation, out_filename), window)


# ---------------------------------------------------
# Main Loop
# ---------------------------------------------------
for file in os.listdir(input_folder):
    if not file.lower().endswith(".csv"):
        continue

    filepath = os.path.join(input_folder, file)

    df = pd.read_csv(filepath, header=0)  # row 1 = header
    columns_to_process = df.columns[0:]   # columns starting at #1 (A)

    for col in columns_to_process:
        if col == "df/dt":
            continue
        series = df[col].iloc[:54000].reset_index(drop=True)

        for state, ranges in all_state_ranges.items():
            for (r_start, r_end) in ranges:
                generate_windows_for_region(
                    series, r_start, r_end, state, file, col,
                    output_folder_normal, n, stride
                )

print("Chunked windows saved to:", output_folder_normal, output_folder_oscillation)
