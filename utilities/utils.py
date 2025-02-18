import random
import time

# Generate a random graph with num_nodes nodes and density probability of edge between any two nodes
def generate_random_graph(num_nodes, density):
    """
    Generates a random graph. (Mostly used for testing purposes)

    :param num_nodes: Number of graph vertices
    :type num_nodes: int
    :param density: Average probability of two nodes being connected
    :type density: float
    :return: The generated graph
    :rtype: Graph
    """
    from graph.base import Graph
    graph = Graph(num_nodes)
    random.seed(time.time())

    for i in range(1, num_nodes):
        for j in range(i):
            if random.random() < density:
                graph.add_edge(i, j)

    return graph

class Log():
    """
    Log class to handle node intermediate clique and coloring logging
    """
    def __init__(self, filepath):
        """
        Constructor.

        :param filepath: path of log file for output
        :type filepath: str
        """
        self.log = ""
        self.filepath = filepath

    def append(self, s):
        """
        Appends a string to the log.

        :param s: any string
        :type s: str
        """
        self.log += s

    def __str__(self):
        """
        Cast Log object to string by returning internal log string.
        
        :return: log string
        :rtype: str
        """
        return self.log

    def printToFile(self, path=""):
        """
        Output log to file.

        :param path: optional file path, that will be saved for future calls
        :type path: str
        """
        if path != "":
            self.filepath = path

        with open(self.filepath, "w") as out:
            out.write(self.log)

# Parse a .col file and return a graph
def parse_col_file(file_path):
    """
    Parses a .col file to construct the Graph corresponding to the instance

    :param file_path: Relative path to .col file
    :type file_path: str
    :return: The Graph object corresponding to instance graph
    :rtype: Graph
    """
    from graph.base import Graph
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

def secondsAsStr(s):
    """
    Parses an amount of seconds to hh:mm:ss format.

    :param s: seconds
    :type s: float
    :return: Time as hh:mm:ss format
    :rtype: str
    """
    h = int(s/60/60)
    m = int(s/60)
    return f"{h}h{m}m{s:.3f}s"

def output_results(instance_name, solver_name, solver_version, num_workers, num_cores, wall_time, time_limit, graph, coloring):
    """
    Output results of an instance to file.

    :param solver_name: Solver description (heuristics used, parallelization library)
    :type solver_name: str
    :param solver_version: Solver version (vM.m.p)
    :type solver_version: str
    :param num_workers: Number of MPI ranks used
    :type num_workers: int
    :param num_cores: Number of cores available to each rank
    :type num_cores: int
    :param wall_time: Total execution time in seconds
    :type wall_time: int
    :param time_limit: Execution time limit in seconds 
    :type time_limit: int
    :param graph: Instance graph
    :type graph: Graph
    :param coloring: List of colors (index is node, value is color) forming a proper coloring of the graph 
    :type coloring: list[int]
    """
    # Get file name without path and extension
    instance_file = instance_name.split('/')[-1].split('.')[0]
    output_file = f"../results/{instance_file}.output"
    
    with open(output_file, 'w') as f:
        f.write(f"problem_instance_file_name: {instance_file}.col\n")
        f.write(f"cmd_line: mpirun -n {num_workers} {solver_name} {instance_file}.col\n")
        f.write(f"solver_version: {solver_version}\n")
        f.write(f"number_of_vertices: {graph.num_nodes}\n")
        f.write(f"number_of_edges: {sum(len(neighbors) for neighbors in graph.adj_list.values())}\n")
        f.write(f"time_limit_sec: {time_limit}\n")
        f.write(f"number_of_worker_processes: {num_workers}\n")
        f.write(f"number_of_cores_per_worker: {num_cores}\n")
        f.write(f"wall_time_sec: {wall_time}\n")
        f.write(f"is_within_time_limit: {wall_time <= time_limit}\n")

        f.write(f"number_of_colors: {len(set(coloring))}\n")
        
        # Write vertex-color assignments
        for vertex in range(graph.num_nodes):
            f.write(f"{vertex+1} {coloring[vertex]}\n")