import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from WorkingChunkedDataLoader20260122 import load_filtered_nparrays

import time

#PyCharm should automatically download imported libraries, if it does not do this, download them manually

#file config
window_length = 512

#Set to None for no limit
nitems = None
overlap_length = 128

#set to VP Angle, VP Magnitude, IP Angle, IP Magnitude or Frequency
parameter_tested = "IP Angle"

desired_labels = ["Normal", "Oscillation"]
out_config_file = "config.txt"
out_folder = f"Z:\\Eversource RF Outputs\\ {parameter_tested} {window_length} Samples {overlap_length} Overlap Variable Depth and Trees"
print(f"Output folder: {out_folder}")

#make out folder directory and get the directory where this python file is stored
os.makedirs(out_folder, exist_ok=True)
folder_dir = os.path.dirname(os.path.realpath(__file__))

base = "Z:\\Chunked Data\\0.25 overlap"

folder_labels = {
    r"Z:\Chunked Data\0.25 overlap\NUMPY Chunked Example Data MinMax Normal 512 Sample Windows 128 Overlap with Backshift": "Normal",
    r"Z:\Chunked Data\0.25 overlap\NUMPY Chunked Example Data MinMax Oscillation 512 Sample Windows 128 Overlap with Backshift": "Oscillation"
}
include_terms = [f"{parameter_tested}"]

#load the nparrays
data_list, state_labels, numfiles, numlabels = load_filtered_nparrays(
    base_dir=base,
    folder_labels=folder_labels,
    include_terms=include_terms,
    n_samples=None,  # auto-determine
    n_items=nitems,
    balance_data=True
 )

X = np.vstack(data_list)
y = state_labels
print(X)
print(y)
print(f"XShape: {X.shape}")

encoder = LabelEncoder()
Y = encoder.fit_transform(state_labels)
print("Encoded List", Y)
print("Length of Encoded List", len(Y))

# ============= Training - add loops to this for different variables as needed =============
j = 1
i = 1
while i <= 50:
    j = 1
    while j <= 50:

        start_time=time.time()

        # Random forest config:
        test_size = 0.2
        random_state = 42
        n_estimators = i
        max_depth = j
        max_features = 'sqrt'
        n_jobs = -1

        #should rename this
        out_file = f"classification_report_eversource_normal_oscillation_{n_estimators}e_{max_depth}d_{random_state}r"

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        #Create the random forest classifier
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    max_features=max_features, random_state=random_state, n_jobs=n_jobs)
        rf.fit(X_train, y_train)

        #predict using the test data
        y_predict = rf.predict(X_test)
        y_predict_flat = y_predict.flatten()
        y_test_flat = y_test.flatten()

        y_predict_flat_decoded = encoder.inverse_transform(y_predict_flat)
        y_test_flat_decoded = encoder.inverse_transform(y_test_flat)

        report = classification_report(y_test_flat_decoded, y_predict_flat_decoded, zero_division=0)
        #print(report)

        #Confusion matrix. Prints to the console and displays

        ConfMatrix=confusion_matrix(y_test_flat_decoded, y_predict_flat_decoded)
        #print(ConfMatrix)
        """
        #print("Confusion Matrix", ConfMatrix)
        disp = ConfusionMatrixDisplay(confusion_matrix=ConfMatrix,display_labels=desired_labels)

        #Conf matrix display, commented so it doesn't pause the program
        disp.plot()
        plt.title(f"Confusion Matrix, RF classifier, Eversource Example Data {include_terms},\n n_estimators={n_estimators}, test_size={test_size}, max_depth={max_depth}, random_state={random_state}")

        #get the end of the RF runtime in seconds
        #this will change depending on the hardware. Vast majority was computed on a laptop
        #Ryzen 7 5800U with 24GB 3200MHz DDR4 memory
        

        #confusion matrix PNG
        CM_path = os.path.join(out_folder, out_file + "_confusion_matrix.png")
        disp.figure_.savefig(CM_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {CM_path}")
        """
        #text confusion matrix
        #this is redundant
        results_path = os.path.join(out_folder, out_file + "_report.txt")

        with open(results_path, "w") as f:
            f.write(classification_report(y_test, y_predict))
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(ConfMatrix))

        print(f"Report saved to {results_path}")

        end_time = time.time()
        total_time = end_time - start_time
        # ============= Metrics for Aggregation =============

        report_dict = classification_report(
            y_test,
            y_predict,
            output_dict=True
        )

        accuracy = accuracy_score(y_test, y_predict)

        run_metrics = {
            "accuracy": accuracy,
            "f1_macro": report_dict["macro avg"]["f1-score"],
            "precision_macro": report_dict["macro avg"]["precision"],
            "recall_macro": report_dict["macro avg"]["recall"],
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "test_size": test_size,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "runtime_sec": total_time,
            "report_txt": results_path,
            "numberf_of_files": numfiles,
            "labels":include_terms


        }

        # ============= Append to CSV (one row per loop) =============
        csv_path = os.path.join(out_folder, f"rf_eversource_{window_length}_{parameter_tested}_results.csv")

        #using a df is slow, should change this
        df = pd.DataFrame([run_metrics])


        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

        print(f"Metrics appended to {csv_path}")

        # ============= save text config file =============
        config_path = os.path.join(out_folder, out_file + "_" + out_config_file)

        with open(config_path, "w") as f:
            f.writelines([
                f"Desired labels: {desired_labels}\n",
                f"Folder labels: {folder_labels}\n",
                f"Report file: {results_path}\n",
                f"Test size: {test_size}\n",
                f"Random state: {random_state}\n",
                f"N estimators: {n_estimators}\n",
                f"Max depth: {max_depth}\n",
                f"Max features: {max_features}\n",
                f"N jobs: {n_jobs}\n",
                f"Runtime (sec): {total_time:.2f}\n"
                f"Conf matrix: {ConfMatrix}"
            ])

        print(f"Config saved to {config_path}")

        j = j + 1
    i = i + 1
