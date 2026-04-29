import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

#This is the basic script that we used for box plotting our data in our report.
#the exact configuration was changed (such as the filepaths) and is rather manual
#but this template is pretty much what we used for all boxplots


#In this example random forests wer trained on Eversource, mixed GESL Oscillation, and mixed GESL switching data
#Testing was done on Eversource data and F1 score was plotted.


Data1_train_Ev_test = [] #Eversource Data
Data2_train_Ev_test = [] #GESL Osc test
Data3_train_Ev_test = [] #GESL Switching test


#Highest common directory for data
base_dir1 = r"Z:\Eversource RF Outputs"
#measurement channels
Freq1 = os.path.join(base_dir1, r"Frequency 1024 Samples 512 Overlap Variable Depth and Trees nparray 04-10-2026\rf_experiment_results.csv")
IPMag1 = os.path.join(base_dir1, r"IP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray 01-30-2026\rf_eversource_1024_IP Magnitude_results.csv")
IPAngle1 = os.path.join(base_dir1, r"IP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
VPMag1 = os.path.join(base_dir1, r"VP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray 02-10-2026\rf_eversource_1024_VP Magnitude_results.csv")
VpAngle1 = os.path.join(base_dir1, r"VP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray test\rf_experiment_results.csv")
filelocation_arr1 = [Freq1, IPMag1, IPAngle1, VPMag1, VPMag1]

#GESL Oscillation
base_dir2 = r"Z:\EversourceGESL RF Outputs\OscillationNormal"
Freq2 = os.path.join(base_dir2, r"Frequency 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
IPMag2 = os.path.join(base_dir2, r"IP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
IPAngle2 = os.path.join(base_dir2, r"IP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
VPMag2 = os.path.join(base_dir2, r"VP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
VpAngle2 = os.path.join(base_dir2, r"VP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")

filelocation_arr2 = [Freq2, IPMag2, IPAngle2, VPMag2, VPMag2]

# GESL switching
base_dir3 = r"Z:\EversourceGESL RF Outputs\SwitchingNormal"
Freq3 = os.path.join(base_dir3, r"Frequency 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
IPMag3 = os.path.join(base_dir3, r"IP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
IPAngle3 = os.path.join(base_dir3, r"IP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
VPMag3 = os.path.join(base_dir3, r"VP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
VpAngle3 = os.path.join(base_dir3, r"VP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
filelocation_arr3 = [Freq3, IPMag3, IPAngle3, VPMag3, VPMag3]

for location in filelocation_arr1:
    df = pd.read_csv(location) #read each file path in the array
    Data1_train_Ev_test.append(df["f1_macro"].tolist()) #Looks for F1 score
    print(Data1_train_Ev_test) #for debug

for location in filelocation_arr2:
    df = pd.read_csv(location) #read each file path in the array
    Data2_train_Ev_test.append(df["f1_macro"].tolist())
    print(Data2_train_Ev_test)

for location in filelocation_arr3:
    df = pd.read_csv(location) #read each file path in the array
    Data3_train_Ev_test.append(df["f1_macro"].tolist())
    print(Data3_train_Ev_test)

print("Data loaded, plotting")

fig, ax = plt.subplots()

#positions of each boxplot, change this for offsets
positions1 = [1, 2, 3, 4, 5] #Eversource

positions2 = [1, 2, 3, 4, 5] #GESL Oscillation

positions3 = [1, 2, 3, 4, 5] #GESL Switching

#Plot each set of boxes
box1 = ax.boxplot(Data1_train_Ev_test, positions=positions1, widths=0.3, patch_artist=True, showmeans=True, showfliers=False)
box2 = ax.boxplot(Data2_train_Ev_test, positions=positions2, widths=0.3, patch_artist=True, showmeans=True, showfliers=False)
box3 = ax.boxplot(Data3_train_Ev_test, positions=positions3, widths=0.3, patch_artist=True, showmeans=True, showfliers=False)

# Color them differently
for b in box1['boxes']: # Eversource
    b.set_facecolor('lightgreen')

for b in box2['boxes']: #GESL Oscillation
    b.set_facecolor('lightcoral')

for b in box3['boxes']: # GESL Switching
    b.set_facecolor('lightblue')

#change legend names, add coloring
legend_handles = [
    mpatches.Patch(color='lightgreen', label='Eversource Data'),
    mpatches.Patch(color='lightcoral', label='GESL Oscillation Data'),
    mpatches.Patch(color='lightblue', label='GESL Switching Data')
]

ax.legend(handles=legend_handles)

# Set x-axis labels
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['Frequency', 'IP Magnitude', 'IP Angle', 'VP Magnitude', 'VP Angle'])
plt.title("F1 Scores of Eversource Trained Models")

plt.show()