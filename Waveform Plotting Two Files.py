import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_data(file_path1, file_path2, x_column, y_column):
    try:
        # Load the dataset
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)

        # Plotting configuration
        plt.figure(figsize=(10, 6))
        plt.plot(df1[x_column], df1[y_column], linestyle='-', color='b', label="Normal State")
        plt.plot(df2[x_column], df2[y_column], linestyle='-', color='r', label="Oscillation State")

        # Adding labels and styling
        plt.title(f'Eversource Min-Max Normalized {y_column} vs {x_column}', fontsize=14)
        plt.xlabel(f'{x_column} (s)', fontsize=12)
        plt.ylabel(f'{y_column} (V)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylim(0,1)

        # Show the plot
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Error: The file was not found. Please check the file path.")
    except KeyError:
        print(f"Error: One of the columns ('{x_column}' or '{y_column}') does not exist in the CSV.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
# plot_csv_data('your_data.csv', 'Time', 'Voltage')

filepath1 = r"Z:"
filepath2 = r"Z:"
x_column = 'Time'
y_column = 'Voltage Magnitude'
plot_csv_data(filepath1, filepath2, x_column, y_column)