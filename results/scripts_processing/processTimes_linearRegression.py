import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import linregress
import sys

# Define the path to the instances folder
instances_folder = "../../instances/"

def read_table(file_path):
    instances = []
    wall_times = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(";")
            if len(parts) < 6 or "instance1" in parts[0]:
                continue  # Skip header or invalid lines
            if parts[3].strip() != "Yes":
                continue  # Exclude instances that are not optimal
            instances.append(parts[0].strip())
            wall_times.append(int(parts[4].strip().split()[0]))  # Extract wall time in seconds
    return instances, wall_times

def get_instance_size(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("p edge"):
                return int(line.split()[2])  # Extract the first numerical value after "p edge"
    return None

# Read data from table file
table_file = sys.argv[1]  
instances, wall_times = read_table(table_file)

# Get instance sizes
instance_sizes = [get_instance_size(os.path.join(instances_folder, instance)) for instance in instances]

# Filter out missing instances
filtered_data = [(size, wt) for size, wt in zip(instance_sizes, wall_times) if size is not None]

# Aggregate data to remove duplicates
size_to_times = defaultdict(list)
for size, time in filtered_data:
    size_to_times[size].append(time)

unique_sizes = sorted(size_to_times.keys())
avg_times = [np.mean(size_to_times[size]) for size in unique_sizes]

# Perform linear regression to get trend line
slope, intercept, _, _, _ = linregress(unique_sizes, avg_times)
trend_line = [slope * x + intercept for x in unique_sizes]

# Plot scatter with trend line
plt.figure(figsize=(10, 5))
plt.scatter(unique_sizes, avg_times, color='blue', label='Instances')
plt.plot(unique_sizes, trend_line, linestyle='-', color='red', label='Trend Line')
plt.xlabel("Number of Nodes")
plt.ylabel("Wall Time (sec)")
plt.title("Wall Time vs. Number of Nodes in Instance")
plt.legend()
plt.grid(True)
plt.show()