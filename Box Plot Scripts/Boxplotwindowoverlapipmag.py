import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import os


Data1_train_Ev_test = [] #Eversource Data
Data2_train_Ev_test = [] #GESL Osc test
Data3_train_Ev_test = [] #GESL Switching test

#50% Overlap
#data's base location; Highest common directory
base_dir1 = r"Z:\Eversource RF Outputs"
#ALL OF THESE NEED TO GET CHANGED FOR PLOTTING NEW GRAPH OR IT WILL BE WRONG
Freq1 = os.path.join(base_dir1, r"IP Magnitude 1024 Samples 512 Overlap Variable Depth and Trees nparray 01-30-2026\rf_eversource_1024_IP Magnitude_results.csv")
IPMag1 = os.path.join(base_dir1, r"IP Magnitude 512 Samples 256 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")
IPAngle1 = os.path.join(base_dir1, r"IP Magnitude 256 Samples 128 Overlap Variable Depth and Trees nparray 02-03-2026 2\rf_eversource_256_IP Magnitude_results.csv")
#VPMag1 = os.path.join(base_dir1, r"Frequency 128 Samples 64 Overlap Variable Depth and Trees nparray test\rf_eversource_128_Frequency_results.csv")

filelocation_arr1 = [Freq1, IPMag1, IPAngle1]
#freq1 = 1024, IPMag1 = 512, IPAngle1 = 256, VPMag1 = 128

#25% overlap
base_dir2 = r"Z:\Eversource RF Outputs"

Freq2 = os.path.join(base_dir2, r"IP Magnitude 1024 Samples 256 Overlap Variable Depth and Trees, Item limit=None 02-19-2026\rf_eversource_1024_IP Magnitude_results.csv")
IPMag2 = os.path.join(base_dir2, r"IP Magnitude 512 Samples 128 Overlap Variable Depth and Trees, Item limit=None 02-19-2026\rf_eversource_512_IP Magnitude_results.csv")

filelocation_arr2 = [Freq2, IPMag2]
#Freq2 = 1024, IPMag2 = 512

#75% Overlap
# GESL SWITCHING usually
base_dir3 = r"Z:\Eversource RF Outputs"

Freq3 = os.path.join(base_dir3, r"IP Magnitude 1024 Samples 768 Overlap Variable Depth and Trees, Item limit=None 02-18-2026\rf_eversource_1024_IP Magnitude_results.csv")
IPMag3 = os.path.join(base_dir3, r"IP Magnitude 512 Samples 384 Overlap Variable Depth and Trees nparray\rf_experiment_results.csv")

filelocation_arr3 = [Freq3, IPMag3]
#Freq3 = 1024, IPMag3 = 512

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


fig, ax = plt.subplots()

positions1 = [1, 2, 3]

positions2 = [1, 2]

positions3 = [1, 2]

box1 = ax.boxplot(Data1_train_Ev_test, positions=positions1, widths=0.3, patch_artist=True, showmeans=True, showfliers=False)
box2 = ax.boxplot(Data2_train_Ev_test, positions=positions2, widths=0.3, patch_artist=True, showmeans=True, showfliers=False)
box3 = ax.boxplot(Data3_train_Ev_test, positions=positions3, widths=0.3, patch_artist=True, showmeans=True, showfliers=False)


for b in box1['boxes']:
    b.set_facecolor('#AC2B37')

for b in box2['boxes']:
    b.set_facecolor('#A9B0B7')

for b in box3['boxes']:
    b.set_facecolor('#FEFFBE')

for median in box1['medians']:
    median.set_color('black')

for median in box2['medians']:
    median.set_color('black')

for median in box3['medians']:
    median.set_color('black')



legend_handles = [
    mpatches.Patch(color='#AC2B37', label='50% Overlap'),
    mpatches.Patch(color='#A9B0B7', label='25% Overlap'),
    mpatches.Patch(color='#FEFFBE', label='75% Overlap')
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


ax.legend(handles=legend_handles ) #loc='upper right', bbox_to_anchor=(1.5,1)


ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['1024', '512', '256'])
ax.set_ylabel('F1 Score')
plt.suptitle("F1 scores of Eversource window sizes, IP Magnitude 50%, 25%, 75% Overlap")
#plt.title("1024 window size, 50% overlap", fontsize = 11)

#plt.savefig(r'Z:\BOXPLOTS\Eversource, GESL Osc & Switching, each tested and trained on its own data.png')
#plt.tight_layout()
plt.show()