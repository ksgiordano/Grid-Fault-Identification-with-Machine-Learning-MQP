import os
import numpy as np
#this simple script just takes a specified folder with csvs and saves them as
#numpy arrays in a second specified folder.

n= 512 #window length in samples
overlap = 512//4 #overlap in samples
input_folder = fr"Z:\Chunked Data\0.25 overlap\Chunked Example Data MinMax Oscillation {n} Sample Windows {overlap} Overlap with Backshift"
output_folder = fr"Z:\Chunked Data\0.25 overlap\NUMPY Chunked Example Data MinMax Oscillation {n} Sample Windows {overlap} Overlap with Backshift"
delimiter = ","  #tells script what to use as a delimiter, comma for CSV


os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".csv"):
        csv_path = os.path.join(input_folder, filename)
        npy_path = os.path.join(
            output_folder,
            os.path.splitext(filename)[0] + ".npy"
        )

        data = np.genfromtxt(
            fname=csv_path,
            delimiter=delimiter,
            dtype=float
        )
        np.save(npy_path, data)
        print(f"Saved: {npy_path}")
