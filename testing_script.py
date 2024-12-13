import os
import subprocess
import pandas as pd

# Define the order of Field of View and Orientation
fov_orient_order = [
    {'Field of View': 360, 'Orientation': 'None'},
    {'Field of View': 270, 'Orientation': 'None'},
    {'Field of View': 180, 'Orientation': 'None'},
    {'Field of View':  90, 'Orientation': 'None'},
    {'Field of View':  70, 'Orientation': 'None'},
    {'Field of View': 360, 'Orientation': 'left'},
    {'Field of View': 360, 'Orientation': 'right'},
    {'Field of View': 360, 'Orientation': 'back'},
    {'Field of View': 270, 'Orientation': 'left'},
    {'Field of View': 270, 'Orientation': 'right'},
    {'Field of View': 270, 'Orientation': 'back'},
    {'Field of View': 180, 'Orientation': 'left'},
    {'Field of View': 180, 'Orientation': 'right'},
    {'Field of View': 180, 'Orientation': 'back'},
    {'Field of View':  90, 'Orientation': 'left'},
    {'Field of View':  90, 'Orientation': 'right'},
    {'Field of View':  90, 'Orientation': 'back'},
    {'Field of View':  70, 'Orientation': 'left'},
    {'Field of View':  70, 'Orientation': 'right'},
    {'Field of View':  70, 'Orientation': 'back'},
]

# Base command template
base_command = "python test.py --dataset CVUSA --data_dir /gpfs2/scratch/xzhang31/CVUSA/dataset --model_path 1733954021_GeoDTR_CVUSA_True_False_test --verbose"

# Prepare a list to store the results
results = []

# Iterate through the specified order
for item in fov_orient_order:
    fov = item['Field of View']
    orient = item['Orientation']
    
    # Generate the command
    command = f"{base_command} --fov {fov} --orient {orient.lower()}"  # Ensure orientation is in lowercase
    print(f"Running command: {command}")
    
    # Run the command and capture the output
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Parse the output metrics
    output = process.stdout.decode("utf-8")
    metrics = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":")
            value = value.strip()
            if value:  # Only convert non-empty values
                metrics[key.strip()] = float(value)
    
    # Store the results
    results.append({
        "Field of View": fov,
        "Orientation": orient,
        "Top 1": metrics.get("top1", None),
        "Top 5": metrics.get("top5", None),
        "Top 10": metrics.get("top10", None),
        "Top 1%": metrics.get("top1%", None),
    })
    
    # Optional: Print progress
    if process.returncode == 0:
        print(f"Command completed successfully for FOV={fov}, Orientation={orient}")
    else:
        print(f"Command failed for FOV={fov}, Orientation={orient}. Return code: {process.returncode}")

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to CSV and Excel files
results_df.to_csv('GeoDTR_Robust_Aug_results.csv', index=False)
results_df.to_excel('GeoDTR_Robust_Aug_results.xlsx', index=False)

print("Results have been saved to 'results.csv' and 'results.xlsx'")
