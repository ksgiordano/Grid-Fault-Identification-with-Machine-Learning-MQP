import pandas as pd
import matplotlib.pyplot as plt
import os
#import numpy as np

samples = 1024


def plotheatmap(path, plottype, nsamples, type, name):
    for fname in os.listdir(path):
        if not fname.lower().endswith(".csv"):
            continue

        filepath = os.path.join(path, fname)
        df = pd.read_csv(filepath)

        if not plottype:
            raise TypeError

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
        plt.imshow(pivot, aspect="auto", origin="lower", cmap="jet",
                   extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(label=name)

        plt.xlabel("Number of Estimators")
        plt.ylabel("Max Depth")
        plt.suptitle(f"RF Eversource Data Normal & Oscillation Classification. {name} vs Max Depth & Number of Estimators.", wrap=True, )
        plt.title(f"{nsamples} samples per window, {type}, test_size = 0.2, randomstate=42", wrap=True,
                  fontsize = 10)
        plt.locator_params(axis='both', nbins=26, tight=True)

        plt.tight_layout()
        name = os.path.join(path, f"heatmap{name}log.png")
        print(name)


        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.savefig(fname=name, format="png")
        plt.show()

plotheatmap(path="Z:\\Eversource RF Outputs\\IP Angle 256 Samples 128 Overlap Variable Depth and Trees nparray 02-03-2026",
            plottype="accuracy",
            nsamples=256,
            type="IP Angle",
            name = "Accuracy"
            )

