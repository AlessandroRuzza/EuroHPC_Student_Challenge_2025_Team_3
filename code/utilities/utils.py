import random
import time
import argparse
import importlib
import inspect
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent

algorithms_dir = current_dir.parent / 'algorithms'
graph_dir = current_dir.parent / 'graph'
sys.path.append(str(graph_dir.parent))
sys.path.append(str(algorithms_dir.parent))
from graph.base import Graph

from algorithms.maxclique_heuristics import *
from algorithms.coloring_heuristics import *
from algorithms.branching_strategies import *

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
        ## @var log
        # Log string
        self.log = ""
        ## @var filepath
        # Path of log file
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
    :raises FileNotFoundError: If the specified file is not found
    :raises IOError: If there is an error reading the file.
    :return: The Graph object corresponding to instance graph
    :rtype: Graph
    """

    graph = None

    try:
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
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified file '{file_path}' was not found.")
    except IOError as e:
        raise RuntimeError(f"An error occurred while reading the file '{file_path}': {e}")


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

def output_results(instance_name, output_folder, solver_name, solver_version, num_workers, num_cores, wall_time, time_limit, graph, coloring, maxCliqueSize):
    """
    Output results of an instance to file.

    :param instance_name: Instance file name
    :type instance_name: str
    :param output_folder: Folder in which to write the results
    :type output_folder: str 
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
    :param maxCliqueSize: Best lower bound found
    :type maxCliqueSize: int
    """
    # Get file name without path
    instance_file = instance_name.split('/')[-1][0:-4]
    output_file = f"{output_folder}/output/{instance_file}.output"
    
    with open(output_file, 'w') as f:
        f.write(f"problem_instance_file_name: {instance_file}.col\n")
        f.write(f"cmd_line: mpirun -n {num_workers} MPI/mpi.py {instance_name} {output_folder} --cpusPerTask {num_cores}\n")
        f.write(f"solver_version: {solver_version}\n")
        f.write(f"number_of_vertices: {graph.num_nodes}\n")
        f.write(f"number_of_edges: {sum(len(neighbors) for neighbors in graph.adj_list.values())}\n")
        f.write(f"time_limit_sec: {time_limit}\n")
        f.write(f"number_of_worker_processes: {num_workers}\n")
        f.write(f"number_of_cores_per_worker: {num_cores}\n")
        f.write(f"wall_time_sec: {wall_time}\n")
        f.write(f"is_within_time_limit: {wall_time < time_limit}\n")

        f.write(f"number_of_colors: {len(set(coloring))}\n")
        f.write(f"max_clique_size: {maxCliqueSize}\n")
        
        # Write vertex-color assignments
        for vertex in range(graph.num_nodes):
            f.write(f"{vertex+1} {coloring[vertex]}\n")


def load_heuristics(module_name):
    """Dynamically loads all heuristic classes from a file
    
    :param module_name: name of the file
    :type module_name: str
    :return: dictionary containing the heuristics, keys show their names, the values their respective objects
    :rtype: dict[str:obj]"""

    module = importlib.import_module(module_name)
    heuristics = {}

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if ((hasattr(obj, "find_pair") or  hasattr(obj, "find_max_clique") or hasattr(obj, "find_coloring"))  # Valid classes reference
            and not inspect.isabstract(obj)): # Remove abstract classes
            heuristics[name] = obj  

    return heuristics

def instantiate(heuristics, name, default, args):
    """
    Instantiates the heuristics. If none are specified, it chooses the default one. Adds num_workers for parallel ones

    :param heuristics: dict containing all heuristics in a key:value format, check load_heuristics for details
    :type heuristics: dict[str:obj]
    :param name: name of the class
    :type name: str
    :param default: default class to return if name is None
    :type param: obj
    :param args: the args object
    :type args: Namespace
    :return: instance of object
    :rtype: obj
    """

    # Setting up default parameters
    if not name:
        return default

    selected = heuristics[name]
    if "parallel" in name.lower():
        return selected(num_workers=args.cpusPerTask)
    else:
        return selected()

def get_args():
    """
    Parser for the command line arguments
    Returns a Namespace object args, containing:
    - in args.instance: the instance file of the graph
    - in args.outFolderPath: the output folder for the results
    - in args.branch: the Branching Strategy
    - in args.color: the Coloring Heuristic
    - in args.clique: the Max Clique Heuristic

    :raises ValueError: If an invalid heuristic is specified that is not in the available choices.
    :return: Namespace data structure containing info about file and heuristics specified, check description for details
    :rtype: Namespace

    
    """
    parser = argparse.ArgumentParser(description="Chasing the Perfect Hue: A High-Performance Dive into Graph Coloring\n\nA Graph Coloring Solver utilizing the Branch-and-Bound Framework",
        formatter_class=argparse.RawTextHelpFormatter  ,
        epilog=""" 
Example Usage:
--------------
mpi.py ../instances/anna.col ../results/2h_test_output/ --cpusPerTask 16 --branch SaturationBranchingStrategy --color ParallelBacktrackingDSatur --clique ParallelDLS
"""
    )

    parser.add_argument("instance", type=str, help="Graph instance file (utilizing .col format)")
    parser.add_argument("outFolderPath", type=str, help="Output folder of the execution")

    # Heuristic Parallelisation parameter 
    parser.add_argument("--cpusPerTask", type=int, default=8, help="Number of threads to use for each worker, ignored for non-parallel heuristics")

    # Heuristic parameters
    branch_heuristics = load_heuristics("algorithms.branching_strategies")
    parser.add_argument("--branch", type=str, choices=list(branch_heuristics.keys()), 
                        help=f"Branching strategy (default: Saturation)")

    color_heuristics = load_heuristics("algorithms.coloring_heuristics")
    parser.add_argument("--color", type=str, choices=color_heuristics.keys(), 
                        help=f"Coloring heuristic (default: ParallelBacktrackingDSatur)")
    

    clique_heuristics = load_heuristics("algorithms.maxclique_heuristics")
    parser.add_argument("--clique", type=str, choices=clique_heuristics.keys(), 
                        help="Clique finding method (default: ParallelDLS)")

    # Parse arguments
    args = parser.parse_args()
    
    args.branch = instantiate(branch_heuristics, args.branch, SaturationBranchingStrategy(), args)
    
    args.color = instantiate(color_heuristics, args.color, ParallelBacktrackingDSatur(num_workers=args.cpusPerTask), args)

    args.clique = instantiate(clique_heuristics, args.clique, ParallelDLS(num_workers=args.cpusPerTask), args) 

    return args
