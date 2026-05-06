import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#file config
desired_labels = ["Change", "Steady"]
sig_prefix = "sigId"
i = 1500
data_suffix = ".csv"
meta_suffix = "-Metadata.csv"
sig_folder = r"Z:\\GESL PMU Data\GESL Data Provider 9 PMU Only"
out_file = "Provider9-1500-MinMaxNorm.csv"
out_folder = r"Z:\\MinMaxNormDataFiles"

#make out folder directory and get the directory where this python file is stored
os.makedirs(out_folder, exist_ok=True)
folder_dir = os.path.dirname(os.path.realpath(__file__))

data_file = os.path.join(folder_dir, sig_folder, f"{sig_prefix}-{i}{data_suffix}")

#Read the data file
df = pd.read_csv(data_file)
#Get all numbers not in column 1 because that is a time column
num_df = df.iloc[:,1:].select_dtypes(include=['number'])

data = np.array(num_df)
print(data)
rows, cols = data.shape
print(rows, cols)

dataNorm = np.empty((rows, cols))

for i in range (1, cols):
    scaler = MinMaxScaler()
    columnNorm = scaler.fit_transform(data[:, [0, i]])
    print("Column----", columnNorm)
    dataNorm[:,i-1] = columnNorm[:,1]

print("Total Array", dataNorm)
#Save normalized data in csv file
full_path = os.path.join(out_folder, out_file)
np.savetxt(full_path, dataNorm, delimiter=',', fmt='%f')

print(f"Array saved to: {full_path}")