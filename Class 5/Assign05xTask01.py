import os
import glob
import shutil
import pandas as pd
# Move all CSV files to a backup folder
csv_files = glob.glob("*.csv_files/")
for file in csv_files:
    shutil.move(file, "backup_folder/")
    print(f"Moved file: {file}")
# Automating Export
def export_data(df, filename, format):
    if format == "csv":
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename} in CSV format.")
    elif format == "json":
        df.to_json(filename, orient="records")
        print(f"Data exported to {filename} in JSON format.")
    else:
        print("Unsupported format.")
# Example usage:
# Creating a sample dataframe
data = {'Name': ['Alice', 'Bob', 'Charlie'],
'Age': [25, 30, 35],
'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
# Exporting to CSV
export_data(df, "output.csv", "csv")
# Exporting to JSON
export_data(df, "output.json", "json")
