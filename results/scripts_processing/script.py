from os import listdir, stat
from os.path import isfile, join

"""
instance.output file format:

problem_instance_file_name: fpsol2.col
cmd_line: mpirun -n 4 MPI_BacktrackingDSatur_DLS fpsol2.col
solver_version: v1.0.1
number_of_vertices: 496
number_of_edges: 23308
time_limit_sec: 10000
number_of_worker_processes: 4
number_of_cores_per_worker: 2
wall_time_sec: 29
is_within_time_limit: True
number_of_colors: 65
max_clique_size: 65
1 0
2 1
"""
results_root = "../"
instance_root = "../../instances/"

output_files = listdir(results_root)

# Complete paths
output_files = [join(results_root, f) for f in output_files if isfile(join(results_root, f))]

def read_line_value(file, width):
    return file.readline().split(": ")[1].strip('\n').center(width)

optimal_count = 0
instanceCount = 0
final_lines = ["instance1".center(15) + " ; nodes, edges ; ( UB,  LB) ; isOptimum ; wall_time ; nTasks, cpusPerTask\n",]
for output in output_files:
    with open(output, "r") as file:
        instanceName = read_line_value(file, 15)
        instanceCount += 1
        file.readline() # Skip cmd_line
        file.readline() # Skip solver_version
        numVertices = read_line_value(file, 5)
        numEdges = read_line_value(file, 5)
        file.readline() # Skip time_limit_sec
        nTasks = read_line_value(file, 6) # number_of_worker_processes
        coresPerTask = read_line_value(file, 5) # number_of_cores_per_worker
        wall_time = read_line_value(file, 6) # wall_time_sec
        timedOut = read_line_value(file, 0) # is_within_time_limit
        chromatic_number = read_line_value(file, 3) # number_of_colorsmax_clique_size
        max_clique_size = read_line_value(file, 3) # max_clique_size

        # Processing
        isOptimal = "Yes" if timedOut == "True" else "Unknown"
        if isOptimal == "Yes": optimal_count += 1
        isOptimal = isOptimal.center(9)
        
        final_lines.append(f"{instanceName} ; {numVertices}, {numEdges} ; ({chromatic_number}, {max_clique_size})")
        final_lines.append(f" ; {isOptimal} ; {wall_time}sec ; {nTasks}, {coresPerTask}\n")
        """
        Result table format:

        instance1 ; numNodes,edges ; (UB, LB) ; isOptimum ; wall_time ; nTasks, cpusPerTask
        instance2 ; numNodes,edges ; (UB, LB) ; isOptimum ; wall_time ; nTasks, cpusPerTask
        ...
        """

result_postprocessed_output = "./result_table.out"
with open(result_postprocessed_output, "w") as postProcess_out:
    postProcess_out.writelines(final_lines)

print(f"Optimal instances solved: {optimal_count}/{instanceCount}  (total number of instances = {len(listdir(instance_root))})")
print("Missing: ")

instList = set(s.split(".")[0] for s in listdir(instance_root))
solvedList = set(s.split("/")[-1].split(".")[0] for s in output_files)

diff = instList - solvedList
print(diff)

