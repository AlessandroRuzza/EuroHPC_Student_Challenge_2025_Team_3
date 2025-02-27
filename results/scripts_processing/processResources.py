import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import linregress

# Define the path to the tables folder
tables_folder = "./"

def read_tables(folder_path):
    resource_to_times = defaultdict(list)
    
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".table"):
            continue  # Skip non-table files
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(";")
                if len(parts) < 6 or "instance1" in parts[0]:
                    continue  # Skip header or invalid lines
                if parts[3].strip() != "Yes":
                    continue  # Exclude instances that are not optimal
                
                wall_time = int(parts[4].strip().split()[0])  # Extract wall time in seconds
                n_tasks = int(parts[5].strip().split(",")[0].strip())
                cpus_per_task = int(parts[5].strip().split(",")[1].strip())
                resource_key = (n_tasks, 1)
                resource_to_times[resource_key].append(wall_time)
                print(f"Accepted instance: nTasks={n_tasks}, cpusPerTask={cpus_per_task}, wall_time={wall_time}")
    
    return resource_to_times

# Read data from multiple table files
resource_to_times = read_tables(tables_folder)

# Compute average wall time for each resource configuration
resource_keys = sorted(resource_to_times.keys())
avg_times = [np.mean(resource_to_times[key]) for key in resource_keys]
n_tasks_values = [key[0] for key in resource_keys]

# Perform linear regression to get trend line
slope, intercept, _, _, _ = linregress(n_tasks_values, avg_times)
trend_line = [slope * x + intercept + 100 for x in n_tasks_values]

# Scatter plot of resource usage vs. average wall time
plt.figure(figsize=(10, 5))
plt.scatter(n_tasks_values, avg_times, color='blue')
plt.plot(n_tasks_values, trend_line, linestyle='-', color='red', label='Trend Line')
plt.xlabel("Resource Usage (nTasks / cpusPerTask)")
plt.ylabel("Average Wall Time (sec)")
plt.title("Wall Time vs. Resource Usage")
plt.legend()
plt.grid(True)
plt.show()
