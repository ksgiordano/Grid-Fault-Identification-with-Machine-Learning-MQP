import pandas as pd
import matplotlib.pyplot as plt
import os
#import numpy as np
traindata = "Oscillation" #Oscillation or Switching
testdata = "Eversource" #GESL Oscillation, GESL Switching, or Eversource
samples = 1024 #1024 or 512
mtype = "Frequency" #Frequency, VP Angle, VP Magnitude, IP Angle, IP Magnitude

def plotheatmap(path, plottype, nsamples, type, name):
    for fname in os.listdir(path):
        if not fname.lower().endswith(".csv"):
            continue

        filepath = os.path.join(path, fname)
        df = pd.read_csv(filepath)

        if not plottype:
            raise TypeError
        #probably a better way to do this but idc
        if plottype == "f1 macro":
            pivot = df.pivot_table(
                values="f1_macro",
                index="max_depth",
                columns="n_estimators",
                aggfunc="mean"
            )
        elif plottype == "runtime":
            pivot = df.pivot_table(
                values="runtime_sec",
                index="max_depth",
                columns="n_estimators",
                aggfunc="mean"
            )
        elif plottype == "precision macro":
            pivot = df.pivot_table(
                values="precision_macro",
                index="max_depth",
                columns="n_estimators",
                aggfunc="mean"
            )
        elif plottype == "recall macro":
            pivot = df.pivot_table(
                values="recall_macro",
                index="max_depth",
                columns="n_estimators",
                aggfunc="mean"
            )
        elif plottype == "accuracy":
            pivot = df.pivot_table(
                values="accuracy",
                index="max_depth",
                columns="n_estimators",
                aggfunc="mean"
            )
        else:
            raise ValueError
            #title = f"RF Eversource Data Normal & Oscillation Classification. Recall vs Max Depth & Number of Estimators."

        #Z = pivot.values
        x = pivot.columns.values
        y = pivot.index.values
        #Z = np.where(Z <= 0, np.nan, Z)


        fig, ax = plt.subplots()
        plt.imshow(pivot, aspect="auto", origin="lower", cmap="jet", vmin=0.2, vmax=1.0,
                   extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(label=name)

        plt.xlabel("Number of Estimators")
        plt.ylabel("Max Depth")
        plt.title(f"{name} vs Max Depth & Number of Estimators.", wrap=True, fontsize = 10)
        plt.suptitle(f"Mixed {traindata} Training, {testdata} Testing, {type}, {nsamples} samples per window, {nsamples*1//2} backshift", wrap=True)
        plt.locator_params(axis='both', nbins=26, tight=True)

        plt.tight_layout()
        name = os.path.join(path, f"0.2-1.0 Scaled Heatmap {name} Log.png")
        print(name)


        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.savefig(fname=name, format="png")
        plt.show()

plotheatmap(path=f"Z:\\Mixed Train Data\\Outputs\\{traindata} - {testdata} Test\\{mtype}\\{mtype} {samples} {samples//2} Overlap Variable Depth and Trees nparray",
            plottype="f1 macro",
            nsamples=samples,
            type=mtype,
            name = "F1 Score"
            )
