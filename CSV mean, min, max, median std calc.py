import os
import pandas as pd

#this script was used to generate mean, min, max, standard deviation from our output files
#this was mostly used in the creation of the numeric tables.
#an entire directory was passed in, and it scans sub directories for files then calculates the statistics
#outputting to a single file makes it easy to compare data across many runs


def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)

        #check if 4 columns exist
        if df.shape[1] < 4:
            return []

        results = []

        #look at the first four columns, that is where we had all of our scoring saved
        for col in df.columns[:4]:
            series = pd.to_numeric(df[col], errors='coerce').dropna()

            if len(series) == 0:
                continue
            #create an calculate the statistics for each column
            stats = {
                "filepath": file_path,
                "column": col,
                "count": len(series),
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std()
            }

            results.append(stats)

        return results

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

#goes through all sub direcories and looks for csvs that contain "results" in their name
#the results text was present in all of our output csvs, even when we changed naming schemes.
def scan_directory(root_dir, output_file, output_dir):
    all_results = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv") and "results" in file.lower():
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                all_results.extend(process_csv(file_path))

    #save the results
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(os.path.join(output_dir, output_file), index=False)
        print(f"\n {output_file}")
        print(f"\nDone! Results saved to: {output_file}")
    else:
        print("No valid data found.")




root_dir = r"Z:\EversourceGESL RF Outputs\OscillationNormal\New VP Magnitude 512 Samples 384 Overlap Variable Depth and Trees nparray" #CHANGE THISS
output_csv = "EV Train & Test VP MAGNITUDE 512 Samples 384 overlap.csv"
output_dir = r"Z:\EversourceGESL RF Outputs\OscillationNormal\New VP Magnitude 512 Samples 384 Overlap Variable Depth and Trees nparray"

scan_directory(root_dir, output_csv, output_dir)