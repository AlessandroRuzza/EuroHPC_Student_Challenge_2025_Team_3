from graph import Graph
import time
import random

# Generate a random graph with num_nodes nodes and density probability of edge between any two nodes
def generate_random_graph(num_nodes, density):
    graph = Graph(num_nodes)
    random.seed(time.time())

    for i in range(1, num_nodes):
        for j in range(i):
            if random.random() < density:
                graph.add_edge(i, j)

    return graph


# Parse a .col file and return a graph
def parse_col_file(file_path):
    graph = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("c "):
                continue  # Ignore comments

            if line.startswith("p "):
                parts = line.split()
                if len(parts) == 4 and parts[0] == "p" and parts[1] == "edge":
                    num_nodes = int(parts[2])
                    graph = Graph(num_nodes)
                continue

            if line.startswith("e "):
                parts = line.split()
                if len(parts) == 3:
                    node1, node2 = int(parts[1])-1, int(parts[2])-1 # files count nodes from 1, we count from 0
                    if node1 != node2:
                        graph.add_edge(node1, node2)
                continue

    return graph

def timeAsStr(timeStruct = time.localtime()):
    return time.strftime('%H:%M:%S')

def secondsAsStr(s):
    h = s/60/60
    m = s/60
    return f"{h}h{m}m{s}s"

def output_results(instance_name, solver_name, solver_version, num_workers, num_cores, wall_time, time_limit, graph, coloring):
    # Get file name without path and extension
    instance_file = instance_name.split('/')[-1].split('.')[0]
    output_file = f"results/{instance_file}.output"
    
    with open(output_file, 'w') as f:
        f.write(f"problem_instance_file_name: {instance_file}.col\n")
        f.write(f"cmd_line: mpirun -n {num_workers} {solver_name} {instance_file}\n")
        f.write(f"solver_version: {solver_version}\n")
        f.write(f"number_of_vertices: {graph.num_nodes}\n")
        f.write(f"number_of_edges: {sum(len(neighbors) for neighbors in graph.adj_list.values()) // 2}\n")
        f.write(f"time_limit_sec: {time_limit}\n")
        f.write(f"number_of_worker_processes: {num_workers}\n")
        f.write(f"number_of_cores_per_worker: {num_cores}\n")
        f.write(f"wall_time_sec: {wall_time}\n")
        f.write(f"is_within_time_limit: {wall_time <= time_limit}\n")
        f.write(f"number_of_colors: {max(coloring) + 1}\n")
        
        # Write vertex-color assignments
        for vertex in range(graph.num_nodes):
            f.write(f"{vertex} {coloring[vertex]}\n")