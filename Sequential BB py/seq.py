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
    start_time = time.time()
    n = len(graph)
    initial_uf = UnionFind(n)
    initial_edges = set()

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
        if node.lb >= best_ub:
            continue
        if current_ub < best_ub:
            print(f"IMPROVED UB! {current_ub} Time: {time.time() - start_time}")
            best_ub = current_ub
        if current_lb > best_lb:
            print(f"IMPROVED LB! {current_lb} Time: {time.time() - start_time}")
            best_lb = current_lb
        if node.lb == best_ub:
            break

        if (time.time() - start_time) > time_limit:
            is_over_time_limit = True
            continue

        u, v = graph.find_pair(node.union_find, node.added_edges)
        if u is None:  # No pair found
            continue


        # Branch 1: same color (skip if assignment is invalid)
        color_u = node.union_find.find(u)
        color_v = node.union_find.find(v)
        doBranch1 = True
        
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
    graph = parse_col_file(filename)

    # Configuration parameters
    solver_name = "sequential_DSatur_DLS_DegreeBranching"
    solver_version = "v1.0.1"

    # Set up algorithms
    graph.set_coloring_algorithm(DSatur())
    graph.set_clique_algorithm(DLSIncreasingPenalty())
    graph.set_branching_strategy(SaturationBranchingStrategy())

    # MPI parameters
    num_workers = 1
    num_cores = 1

    start_time = time.time()
    best_lb, best_ub, bestColoring, isOverTimeLimit = branch_and_bound(graph, timeLimit)
    wall_time = int(time.time() - start_time)

    isValid = graph.validate(bestColoring)

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
    isValid = graph.validate(bestColoring)

    print(f"Is valid? {isValid}")
    print(f"Passed time limit? {isOverTimeLimit}")

    return isValid and not isOverTimeLimit

def main():
    instance_root = "../instances/"

    # Manually specify instances
    instances = ["queen5_5.col",]
    # Or run all instances in the folder
    # instances = listdir(instance_root)

    # Complete path of instances
    instance_files = [join(instance_root, f) for f in instances if isfile(join(instance_root, f))]
    # Sort by file size (bigger graphs take more time)
    instance_files = sorted(instance_files, key=lambda f: (stat(f).st_size))

    badInstances = ("myciel",)
    for bad in badInstances:
        instance_files = [f for f in instance_files if not f.startswith(instance_root + bad)]

    # longest instances: 
    #   fpsol2  (27.7s)
    #   inithx  (>10k s)
    #   queen*  (> s)
    #   all "myciel*" instances solved to optimality, but the lb is never improved (due to maxClique = 2, chromatic number > 2)
    #       so they keep running until the time limit (or all possible combinations are assigned in nodes)
    #   

    # Idea? Order branch queue by (ub - lb) instead of just ub ?

    print(f"Starting at: {time.strftime('%H:%M:%S', time.localtime())}\n")

    #  To clear instances solved
    # os.remove("solved_instances.txt") 

    start_from_idx = 0
    delay = 2
    timeLimit = 10000

    instance_files = instance_files[start_from_idx::]
    i = start_from_idx+1

    with open(f"solved_instances_time_{timeLimit}.txt", "w") as out:
        for instance in instance_files:
            print(f"Solving {instance}... #{i}/{len(instance_files)}")
            start = time.time()
            optimal = solve_instance(instance, timeLimit)
            elapsed = time.time() - start
            print(f"Time elapsed: {elapsed:.3f}s")

            print(f"Waiting {delay} secs to show result.")
            time.sleep(delay)
            print()
            i+=1

            if optimal:
                out.write(instance + "\n")

if __name__ == "__main__":
    main()


