import os
import numpy as np
from collections import Counter, defaultdict
import random
import time

def load_filtered_nparrays(
    base_dir,
    folder_labels,
    include_terms=None,
    n_samples=None,
    balance_data=False,
    n_items=None,
    random_seed=int(round(time.time()))
):

    if include_terms is None:
        include_terms = []

    random.seed(random_seed)

    raw_data = []
    raw_labels = []
    raw_lengths = []

    #Iterate through specified folders
    for folder_name, state_label in folder_labels.items():
        folder_path = os.path.join(base_dir, folder_name)
        print(folder_path)
        #if cant find the path be sad about it (skip
        if not os.path.isdir(folder_path):
            print(f"[WARN] Folder not found: {folder_path}")
            continue
        #for every file in the current folder
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".npy"):
                continue

            #if the file does not have the included term in its name go to the next file
            if any(term.lower() not in fname.lower() for term in include_terms):
                continue
            #make the filepath for the file
            file_path = os.path.join(folder_path, fname)
            #try to load the file
            try:
                arr = np.load(file_path)
            except Exception as e:
                print(f"[WARN] Failed to load {file_path}: {e}")
                continue

            #make sure that the array is 1d
            arr = np.asarray(arr).ravel()

            #if any numbers in the loaded file are nan then move to the next one
            if np.isnan(arr).any():
                continue

            #append to array, create a list of arrays pretty much
            raw_data.append(arr)
            raw_labels.append(state_label)
            raw_lengths.append(len(arr))
        print(folder_path)
    if len(raw_data) == 0:
        raise ValueError("No valid .npy files found.")

    #determine the sample length
    if n_samples is None:
        n_samples = Counter(raw_lengths).most_common(1)[0][0]

    filtered_data = []
    filtered_labels = []

    for arr, lbl in zip(raw_data, raw_labels):
        if len(arr) >= n_samples:
            filtered_data.append(arr[:n_samples])
            filtered_labels.append(lbl)

    #balancing the dataset
    if balance_data:
        class_buckets = defaultdict(list)
        for arr, lbl in zip(filtered_data, filtered_labels):
            class_buckets[lbl].append(arr)
            
        #filters for n items
        n_classes = len(class_buckets)
        if n_items is not None:
            per_class = n_items // n_classes
        else:
            per_class = min(len(v) for v in class_buckets.values())

        balanced_data = []
        balanced_labels = []

        for lbl, arrs in class_buckets.items():
            take = min(per_class, len(arrs))
            selected = random.sample(arrs, take)
            balanced_data.extend(selected)
            balanced_labels.extend([lbl] * take)

        filtered_data = balanced_data
        filtered_labels = balanced_labels

    num_files = np.size(filtered_data, 0)
    num_labels = np.size(filtered_labels)


    return filtered_data, filtered_labels, num_files, num_labels
