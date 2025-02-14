from mpi4py import MPI
import time
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

from collections import defaultdict
from copy import deepcopy

# MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def branch_and_bound_parallel(graph, time_limit=10000):
    start_time = time.time()
    n = len(graph)
    initial_uf = UnionFind(n)
    initial_edges = set()

    lb = len(graph.find_max_clique(initial_uf, initial_edges))
    ub = len(set(graph.find_coloring(initial_uf, initial_edges)))

    # Shared best upper bound
    best_ub = comm.allreduce(ub, op=MPI.MIN)
    queue = []
    if rank == 0:
        queue.append(BranchAndBoundNode(initial_uf, initial_edges, lb, ub))

    while True:
        if not queue:
            # Check if all processes are done
            done = comm.allreduce(1 if not queue else 0, op=MPI.LAND)
            if done:
                break

        if queue:
            node = queue.pop()

            if node.lb >= best_ub:
                continue

            if node.ub < best_ub:
                best_ub = comm.allreduce(node.ub, op=MPI.MIN)

            if node.lb == best_ub:
                break

            if time.time() - start_time > time_limit:
                break

            u, v = graph.find_pair(node.union_find, node.added_edges)
            if u is None:
                continue

            # Branch 1: Same color
            uf1 = deepcopy(node.union_find)
            uf1.union(u, v)
            edges1 = deepcopy(node.added_edges)
            lb1 = len(graph.find_max_clique(uf1, edges1))
            ub1 = len(set(graph.find_coloring(uf1, edges1)))
            if lb1 < best_ub:
                queue.append(BranchAndBoundNode(uf1, edges1, lb1, ub1))

            # Branch 2: Different color
            uf2 = deepcopy(node.union_find)
            edges2 = deepcopy(node.added_edges)
            ru = uf2.find(u)
            rv = uf2.find(v)
            edges2.add((ru, rv))
            lb2 = len(graph.find_max_clique(uf2, edges2))
            ub2 = len(set(graph.find_coloring(uf2, edges2)))
            if lb2 < best_ub:
                queue.append(BranchAndBoundNode(uf2, edges2, lb2, ub2))

    return best_ub


def solve_instance_parallel(filename, time_limit):
    graph = parse_col_file(filename)

    graph.set_coloring_algorithm(DSatur())
    graph.set_clique_algorithm(DLSIncreasingPenalty())
    graph.set_branching_strategy(SaturationBranchingStrategy())

    start_time = time.time()
    chromatic_number = branch_and_bound_parallel(graph, time_limit)
    wall_time = int(time.time() - start_time)

    if rank == 0:
        print(f"Chromatic number for {filename}: {chromatic_number}")
        print(f"Time: {int(wall_time/60)}m {wall_time%60}s")
        print() # Spacing
        output_results(
            instance_name=filename,
            solver_name="MPI_DSatur_DLS",
            solver_version="v1.0.1",
            num_workers=size,
            num_cores=1,
            wall_time=wall_time,
            time_limit=time_limit,
            graph=graph,
            coloring=graph.find_coloring(UnionFind(len(graph)), set())
        )


def main():
    if rank==0:
        print(f"MPI size = {size}")

    instance_root = "../instances/"

    # Manually specify instances
    instances = ["jean.col", "queen5_5.col", "queen6_6.col", "queen7_7.col"]
    
    # Or run all instances in the folder
    # instances = listdir(instance_root)

    # Complete path of instances
    instance_files = [join(instance_root, f) for f in instances if isfile(join(instance_root, f))]
    # Sort by file size (bigger graphs take more time)
    instance_files = sorted(instance_files, key=lambda f: (stat(f).st_size))

    badInstances = ("myciel") # myciel graphs ub lb never converge (even for optimal ub)
    for bad in badInstances:
        instance_files = [f for f in instance_files if not f.startswith(instance_root + bad)]

    print(f"Starting at: {time.strftime('%H:%M:%S', time.localtime())}\n")
    
    time_limit = 10000

    for instance in instance_files:
        print(f"Solving {instance}...")
        solve_instance_parallel(instance, time_limit)


if __name__ == "__main__":
    main()
