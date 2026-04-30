import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import os

Data1_train_Ev_test = [] #Eversource Data
Data2_train_Ev_test = [] #GESL Osc test
Data3_train_Ev_test = [] #GESL Switching test


#data base location; Highest common directory
base_dir1 = r"Z:\Eversource RF Outputs"
#ALL OF THESE NEED TO GET CHANGED FOR PLOTTING NEW GRAPH OR IT WILL BE WRONG
Freq1 = os.path.join(base_dir1, r"Frequency 1024 Samples 512 Overlap Variable Depth and Trees nparray 04-10-2026\rf_experiment_results.csv")
IPMag1 = os.path.join(base_dir1, r"IP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray 01-30-2026\rf_experiment_results.csv")
IPAngle1 = os.path.join(base_dir1, r"IP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
VPMag1 = os.path.join(base_dir1, r"VP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray 02-10-2026\rf_experiment_results.csv")
VpAngle1 = os.path.join(base_dir1, r"VP Angle 1024 Samples 512 Overlap Variable Depth and Trees nparray test\rf_experiment_results.csv")

filelocation_arr1 = [Freq1, IPMag1, IPAngle1, VPMag1, VpAngle1]

#GESL OSC usually
base_dir2 = r"Z:\Eversource RF Outputs"

Freq2 = os.path.join(base_dir2, r"Frequency 512 Samples 256 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
IPMag2 = os.path.join(base_dir2, r"IP Magnitude 512 Samples 256 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
IPAngle2 = os.path.join(base_dir2, r"IP Angle 512 Samples 256 Overlap Variable Depth and Trees nparray 02-10-2026\rf_experiment_results.csv")
VPMag2 = os.path.join(base_dir2, r"VP Magnitude 512 Samples 256 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
VpAngle2 = os.path.join(base_dir2, r"VP Angle 512 Samples 256 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")

filelocation_arr2 = [Freq2, IPMag2, IPAngle2, VPMag2, VpAngle2]

# GESL SWITCHING usually
base_dir3 = r"Z:\GESL RF Outputs\Switching"

Freq3 = os.path.join(base_dir3, r".f 1024 Samples 512 Overlap Variable Depth and Trees\rf_eversource_1024_.f_results.csv")
IPMag3 = os.path.join(base_dir3, r".ip_m 1024 Samples 512 Overlap Variable Depth and Trees\rf_eversource_1024_.ip_m_results.csv")
IPAngle3 = os.path.join(base_dir3, r".ip_a 1024 Samples 512 Overlap Variable Depth and Trees\rf_eversource_1024_.ip_a_results.csv")
VPMag3 = os.path.join(base_dir3, r".vp_m 1024 Samples 512 Overlap Variable Depth and Trees\rf_eversource_1024_.vp_m_results.csv")
VpAngle3 = os.path.join(base_dir3, r".vp_a 1024 Samples 512 Overlap Variable Depth and Trees\rf_eversource_1024_.vp_a_results.csv")

filelocation_arr3 = [Freq3, IPMag3, IPAngle3, VPMag3, VpAngle3]

for location in filelocation_arr1:
    df = pd.read_csv(location) #read each file path in the array
    Data1_train_Ev_test.append(df["f1_macro"].tolist()) #CHANGE TO NOT F1 MACRO IF YOU WANT A DIFFERENT METRIC
    print(Data1_train_Ev_test)

for location in filelocation_arr2:
    df = pd.read_csv(location) #read each file path in the array
    Data2_train_Ev_test.append(df["f1_macro"].tolist()) #CHANGE TO NOT F1 MACRO IF YOU WANT A DIFFERENT METRIC
    print(Data2_train_Ev_test)

for location in filelocation_arr3:
    df = pd.read_csv(location) #read each file path in the array
    Data3_train_Ev_test.append(df["f1_macro"].tolist()) #CHANGE TO NOT F1 MACRO IF YOU WANT A DIFFERENT METRIC
    print(Data3_train_Ev_test)

print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

fig, ax = plt.subplots()

positions1 = [1 - 0.2, 2 - 0.2, 3 - 0.2, 4 - 0.2, 5 - 0.2]

positions2 = [1 + 0.2, 2 + 0.2, 3 + 0.2, 4 + 0.2, 5 + 0.2]

#positions3 = [1, 2, 3, 4, 5]

box1 = ax.boxplot(Data1_train_Ev_test, positions=positions1, widths=0.4, patch_artist=True, showmeans=True, showfliers=False)
box2 = ax.boxplot(Data2_train_Ev_test, positions=positions2, widths=0.4, patch_artist=True, showmeans=True, showfliers=False)
#box3 = ax.boxplot(Data3_train_Ev_test, positions=positions3, widths=0.5, patch_artist=True, showmeans=True, showfliers=False)


for b in box1['boxes']:
    b.set_facecolor('#AC2B37')

for b in box2['boxes']:
    b.set_facecolor('#A9B0B7')

for median in box1['medians']:
    median.set_color('black')

for median in box2['medians']:
    median.set_color('black')

legend_handles = [
    mpatches.Patch(color='#AC2B37', label='1024 Sample Windows'),
    mpatches.Patch(color='#A9B0B7', label='512 Sample Windows'),
]

median_line = box1['medians'][0]
mean = box1['means'][0]

median_legend = mlines.Line2D([], [], color='black', label='Median')
mean_legend = mlines.Line2D(
    [], [],
    color=mean.get_markeredgecolor(),
    marker=mean.get_marker(),
    linestyle='None',
    markersize=mean.get_markersize(),
    label='Mean'
)


legend_handles.extend([median_legend, mean_legend])

ax.legend(handles=legend_handles)


ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['Frequency', 'IP Magnitude', 'IP Angle', 'VP Magnitude', 'VP Angle'])
ax.set_ylabel('F1 Score')
plt.suptitle("F1 scores of Eversource measurement channels")
plt.title("1024 and 512 window size, 50% overlap", fontsize = 11)

#plt.savefig(r'Z:\BOXPLOTS\Eversource, GESL Osc & Switching, each tested and trained on its own data.png')
#plt.tight_layout()
plt.show()