import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

input_folder = r"Z:\\Example PMU Data - Files with data only"
out_file_suffix = "-MinMaxNorm.csv"
out_folder = r"Z:\\Eversource Data Files MinMax Normalized"

#make out folder directory and get the directory where this python file is stored
os.makedirs(out_folder, exist_ok=True)
folder_dir = os.path.dirname(os.path.realpath(__file__))

for file in os.listdir(input_folder):
    if not file.lower().endswith(".csv"):
        continue

    filepath = os.path.join(input_folder, file)

    #Read the data file
    df = pd.read_csv(filepath, skiprows=1)
    #Get all numbers starting at column 3
    #Previous columns did not contain useful data (dates, times, NaN, etc.)
    num_df = df.iloc[:,3:].select_dtypes(include=['number'])

    data = np.array(num_df)
    #print(data)
    rows, cols = data.shape
    #print(rows, cols)

    dataNorm = np.empty((rows, cols))

    for i in range (cols):
        scaler = MinMaxScaler()
        columnNorm = scaler.fit_transform(data[:, [i]])
        #print("Column----", columnNorm)
        #print(scaler.data_max_)
        dataNorm[:,i] = columnNorm[:,0]

    #print("Total Array", dataNorm)

    #Column Labels
    column_labels = df.iloc[:,3:].columns.tolist()
    #print(column_labels)
    delimiter = ","
    column_labels_string = delimiter.join(column_labels)

    #Save normalized data in csv file
    full_path = os.path.join(out_folder, f"{file}-MinMaxNorm.csv")
    np.savetxt(full_path, dataNorm, delimiter=',', header=column_labels_string, fmt='%f')

    print(f"Array saved to: {full_path}")