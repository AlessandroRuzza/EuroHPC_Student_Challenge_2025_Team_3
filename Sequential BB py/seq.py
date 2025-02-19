import time
import heapq
from copy import deepcopy
from os import listdir, stat
from os.path import isfile, join

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent

utilities_dir = current_dir.parent / 'utilities'
graph_dir = current_dir.parent / 'graph'
algorithms_dir = current_dir.parent / 'algorithms'
sys.path.append(str(utilities_dir.parent))
sys.path.append(str(graph_dir.parent))
sys.path.append(str(algorithms_dir.parent))

from utilities.utils import parse_col_file, output_results

from graph.base import *

from algorithms.maxclique_heuristics import *
from algorithms.coloring_heuristics import *
from algorithms.branching_strategies import *

##########################################################################################

# lb - lower bound
# ub - upper bound
def branch_and_bound(graph, time_limit=10000):
    """
    Implementation of a sequential branch and bound algorithm for the graph coloring problem

    :param graph: graph to solve
    :type graph: Graph
    :param time_limit: time limit in seconds
    :type time_limit: int
    """


    start_time = time.time()
    
    # Initializations
    n = len(graph)
    initial_uf = UnionFind(n)
    initial_edges = set()

    # Initial upper and lower bounds
    lb = len(graph.find_max_clique(initial_uf, initial_edges))
    ub = len(set(graph.find_coloring(initial_uf, initial_edges)))

    best_ub = ub
    best_lb = lb
    queue = []
    heapq.heappush(queue, (ub, BranchAndBoundNode(initial_uf, initial_edges, lb, ub)))

    current_ub, node = None, None
    current_lb = None
    is_over_time_limit = False

    print(f"Starting UB, LB:  {ub, lb}")

    while queue:
        current_ub, node = heapq.heappop(queue)
        current_lb = node.lb

        # Pruning
        if node.lb >= best_ub:
            continue
        
        # Improvements
        if current_ub < best_ub:
            print(f"IMPROVED UB! {current_ub} Time: {time.time() - start_time}")
            best_ub = current_ub
        if current_lb > best_lb:
            print(f"IMPROVED LB! {current_lb} Time: {time.time() - start_time}")
            best_lb = current_lb
        
        # Solution found
        if node.lb == best_ub:
            break
        
        # Time limit reached
        if (time.time() - start_time) > time_limit:
            is_over_time_limit = True
            continue

        ### Branching
        u, v = graph.find_pair(node.union_find, node.added_edges)
        if u is None:  # No pair found
            continue


        # Branch 1: same color
        color_u = node.union_find.find(u)
        color_v = node.union_find.find(v)
        doBranch1 = True
        
        # Do not create branch if assignment is already invalid
        for neighbor in graph.adj_list[u]:
            color_n = node.union_find.find(neighbor)
            if(color_n == color_v):
                doBranch1 = False
                break

        for neighbor in graph.adj_list[v]:
            color_n = node.union_find.find(neighbor)
            if(color_n == color_u or not doBranch1):
                doBranch1 = False
                break

        if doBranch1:
            uf1 = deepcopy(node.union_find)
            uf1.union(u, v)
            edges1 = deepcopy(node.added_edges)

            lb1 = len(graph.find_max_clique(uf1, edges1))
            ub1 = len(set(graph.find_coloring(uf1, edges1)))
            if lb1 < best_ub:
                heapq.heappush(queue, (ub1, BranchAndBoundNode(uf1, edges1, lb1, ub1)))


        # Branch 2: different color
        uf2 = deepcopy(node.union_find)
        edges2 = deepcopy(node.added_edges)
        ru = uf2.find(u)
        rv = uf2.find(v)
        if (ru, rv) not in edges2 and (rv, ru) not in edges2:
            edges2.add((ru, rv))

        lb2 = len(graph.find_max_clique(uf2, edges2))
        ub2 = len(set(graph.find_coloring(uf2, edges2)))
        if lb2 < best_ub:
            heapq.heappush(queue, (ub2, BranchAndBoundNode(uf2, edges2, lb2, ub2)))
        
    bestColoring = graph.find_coloring(node.union_find, node.added_edges)
    return best_lb, best_ub, bestColoring, is_over_time_limit

##########################################################################################

def solve_instance(filename, timeLimit):
    """
    Solve a single instance using a sequential branch and bound algorithm within the time limit

    :param filename: instance to solve
    :type filename: str
    :param timeLimit: time limit in seconds
    :type timeLimit: int
    :return: True if the solution is valid and the time limit was not exceeded, False otherwise
    :rtype: bool
    """
    graph = parse_col_file(filename)

    # Configuration parameters
    solver_name = "sequential_DSatur_DLS_DegreeBranching"
    solver_version = "v1.0.1"

    # Set up heuristics
    graph.set_coloring_algorithm(DSatur())
    graph.set_clique_algorithm(ParallelDLS(dls_instance=DLS()))
    graph.set_branching_strategy(SaturationBranchingStrategy())

    # MPI parameters (not used in this version)
    num_workers = 1
    num_cores = 1

    start_time = time.time()
    best_lb, best_ub, bestColoring, isOverTimeLimit = branch_and_bound(graph, timeLimit)
    wall_time = int(time.time() - start_time)

    isValid = graph.validate_coloring(bestColoring)

    # Output results
    output_results(
        instance_name=filename,
        solver_name=solver_name,
        solver_version=solver_version,
        num_workers=num_workers,
        num_cores=num_cores,
        wall_time=wall_time,
        time_limit=timeLimit,
        graph=graph,
        coloring=bestColoring
    )
    
    if best_lb == best_ub:
        print(f"Optimal solution! Chromatic number = {best_ub}")
    else:
        print(f"Best (LB,UB) = ({best_lb},{best_ub})")

    print(f"Is valid? {isValid}")
    print(f"Passed time limit? {isOverTimeLimit}")
    print(f"Time elapsed: {int(wall_time/60)}m {wall_time%60:.3f}s")

    return isValid and not isOverTimeLimit

def getAllInstances():
    instance_root = "../instances/"
    instances = listdir(instance_root)

    # Complete path of instances
    instance_files = [join(instance_root, f) for f in instances if isfile(join(instance_root, f))]
    # Sort by file size (bigger graphs take more time)
    instance_files = sorted(instance_files, key=lambda f: (stat(f).st_size))

    badInstances = ("myciel",) # myciel graphs ub lb never converge (even for optimal ub)
    for bad in badInstances:
        instance_files = [f for f in instance_files if not f.startswith(instance_root + bad)]

    return instance_files

def main():
    
    if len(sys.argv) < 2:
        print("Usage: python seq.py <instance>")
        sys.exit(1)
        

    instance = sys.argv[1]
    
    print(f"Starting at: {time.strftime('%H:%M:%S', time.localtime())}\n")

    timeLimit = 10000

    print(f"Solving {instance}...")
    solve_instance(instance, timeLimit)

if __name__ == "__main__":
    main()


