from mpi4py import MPI
import time
from os import listdir, stat
from os.path import isfile, join

from threading import Thread, Condition, Event, Lock

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

from copy import deepcopy

# Debug flags
debugSlave = False
debugBounds = True
debugQueue = False

# MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nodesPerSlave = 5

# Branch parameters
worsenTolerance = 1
minQueueLenForPruning = size*2 * nodesPerSlave


def printDebugSlave(str):
    if debugSlave: print(str)

def printDebugBounds(str):
    if debugBounds: print(str)
    
def printConditional(str, condition):
    if condition: print(str)

def branch_node(graph, node):
    u, v = graph.find_pair(node.union_find, node.added_edges)
    if u is None:
        return None
    
    childNodes = []

    # Branch 1: Same color
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
        childNodes.append(BranchAndBoundNode(uf1, edges1, lb1, ub1))

    # Branch 2: Different color
    uf2 = deepcopy(node.union_find)
    edges2 = deepcopy(node.added_edges)
    ru = uf2.find(u)
    rv = uf2.find(v)
    edges2.add((ru, rv))
    lb2 = len(graph.find_max_clique(uf2, edges2))
    ub2 = len(set(graph.find_coloring(uf2, edges2)))
    childNodes.append(BranchAndBoundNode(uf2, edges2, lb2, ub2))

    return childNodes

def handle_slave(graph, slaveRank, 
                 queueLock: Condition, queue: list[BranchAndBoundNode],
                 best_ub_lock: Condition, best_ub, best_lb, best_coloring: list,
                 optimalEvent: Event, timeoutEvent: Event, 
                 start_time, time_limit):
    flag = True
    while True:
        if timeoutEvent.is_set() or optimalEvent.is_set():
            break

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            timeoutEvent.set()
            break

        with queueLock: 
            # Temp debug to validate worsenTolerance pruning strategy
            if not int(elapsed%20) and flag:
                printConditional(f"Len Queue = {len(queue)}", debugQueue)
                flag=False
            elif int(elapsed%20): 
                flag=True

            while not queue:
                queueLock.wait(timeout=1)
                # Timeout to prevent getting stuck when only a few nodes are needed 
                if timeoutEvent.is_set() or optimalEvent.is_set():
                    break

            if timeoutEvent.is_set() or optimalEvent.is_set():
                break
            numNodes = min(len(queue), nodesPerSlave)
            nodes = []
            for _ in range(numNodes):
                nodes.append(queue.pop(0))
        
        pruneNodes = []
        optimalFound = False

        with best_ub_lock:
            for node in nodes:
                if node.ub > best_ub[0]+worsenTolerance and len(queue) > minQueueLenForPruning:  
                    pruneNodes.append(node)
                if node.ub < best_ub[0]:
                    printDebugBounds(f"Slave {slaveRank} improved UB = {node.ub} Time = {int(elapsed/60)}m {elapsed%60:.3f}s")
                    best_coloring.clear()
                    best_coloring.extend(graph.find_coloring(node.union_find, node.added_edges))
                    best_ub[0] = node.ub
                if node.lb > best_lb[0]:
                    printDebugBounds(f"Slave {slaveRank} improved LB = {node.lb} Time = {int(elapsed/60)}m {elapsed%60:.3f}s")
                    best_lb[0] = node.lb
                if best_lb[0] == best_ub[0]:
                    optimalFound = True
                    break

        if optimalFound:
            optimalEvent.set()
            break

        for node in pruneNodes:
            nodes.remove(node)

        if not nodes: continue
        
        # Send (best_ub, nodes) to slave for branching
        comm.send((best_ub[0], nodes), slaveRank)
        childNodes = comm.recv(source=slaveRank)

        if childNodes == "Terminated":
            printDebugSlave(f"Slave Handler {slaveRank} received termination.")
            return
        
        if childNodes is None:
            continue

        with queueLock: 
            for n in childNodes:
                queue.append(n)
            queueLock.notify(len(childNodes))

    # Ensure to kill slave before terminating thread
    comm.send("Kill", slaveRank)
    response = comm.recv(source=slaveRank)
    while response != "Terminated":
        response = comm.recv(source=slaveRank)
    printDebugSlave(f"Slave Handler {slaveRank} received termination.")
    return None

def master_branch_and_bound(graph: Graph, queue: list[BranchAndBoundNode], 
                            best_ub, best_lb, best_coloring, 
                            start_time, time_limit=10000):
    # Run threads that interface with each slave (ranks [1, 2, 3, ..., size-1] )
    slaveHandlers: list[Thread] = []
    queueLock = Condition()
    best_ub_lock = Lock()
    best_ub = [best_ub,]
    best_lb = [best_lb,]
    optimalEvent = Event()
    timeoutEvent = Event()
    for slaveRank in range(1, size):
        slaveHandlers.append(Thread(daemon=True, target=handle_slave, 
                                    args=(graph,slaveRank, queueLock,queue, 
                                          best_ub_lock,best_ub,best_lb,best_coloring, 
                                          optimalEvent,timeoutEvent, 
                                          start_time,time_limit)))

    for slave in slaveHandlers:
        slave.start()

    elapsed = time.time() - start_time
    optimalEvent.wait(timeout=time_limit-elapsed)     # First slave handler that finds an optimal node will notify this lock

    printDebugSlave("Returning, terminating slaves.")
    for slave in slaveHandlers:
        slave.join() # Ensure all threads terminated (implies all slaves terminated as well)

    # Before exiting, check all the generated nodes for improvements
    if timeoutEvent.is_set() and queue:
        best_lb[0] = max(best_lb[0], max(queue, key=lambda n: n.lb).lb)
        best_ub[0] = min(best_ub[0], min(queue, key=lambda n: n.ub).ub)
    
    # Run one thread that solves nodes in this process? (to not waste resources)
    return best_ub[0], best_lb[0], best_coloring

def slave_branch_and_bound(graph):
    while True:
        # Wait for work from master
        nodes = comm.recv(source=0)
        if nodes == "Kill":
            comm.send("Terminated", 0)
            printDebugSlave(f"Slave {rank} sent termination.")
            return
        
        best_ub, nodes = nodes
        graph.best_ub = best_ub
        childNodes = []
        for node in nodes:
            # Run node
            for child in branch_node(graph, node):
                childNodes.append(child)

        # Send back results
        comm.send(childNodes, 0)

def branch_and_bound_parallel(graph, time_limit=10000):
    if rank==0:
        start_time = time.time()
        n = len(graph)
        initial_uf = UnionFind(n)
        initial_edges = set()

        lb = len(graph.find_max_clique(initial_uf, initial_edges))
        cliqueTime = time.time() - start_time
        initial_coloring = graph.find_coloring(initial_uf, initial_edges)
        ub = len(set(initial_coloring))
        colorTime = time.time() - (start_time+cliqueTime)

        # Shared best upper bound
        best_ub = ub
        best_lb = lb
        queue = []

        print(f"Starting (UB, LB) = ({ub}, {lb})")
        print(f"Coloring time = {int(colorTime/60)}m {colorTime%60:.3f}s")
        print(f"Clique time = {int(cliqueTime/60)}m {cliqueTime%60:.3f}s")
        queue.append(BranchAndBoundNode(initial_uf, initial_edges, lb, ub))
        return master_branch_and_bound(graph, queue, best_ub, best_lb, initial_coloring, start_time, time_limit)
    else:
        slave_branch_and_bound(graph)
        return None, None, None

def solve_instance_parallel(filename, time_limit):
    graph = parse_col_file(filename)

    graph.set_coloring_algorithm(BacktrackingDSatur(time_limit=1))
    graph.set_clique_algorithm(ParallelDLS(num_workers=5, lambda_max_steps=10, dls_instance=DLSIncreasingPenalty()))
    graph.set_branching_strategy(SaturationBranchingStrategy())

    start_time = time.time()
    chromatic_number, maxCliqueSize, best_coloring = branch_and_bound_parallel(graph, time_limit)
    wall_time = int(time.time() - start_time)

    if rank == 0:
        print(f"Chromatic number for {filename}: {chromatic_number}")
        if maxCliqueSize < chromatic_number: 
            print(f"MaxClique found = {maxCliqueSize}")
        print(f"Time: {int(wall_time/60)}m {wall_time%60}s")
        print(f"Is Valid? {graph.validate(best_coloring)}")
        if wall_time >= time_limit:
            print("TIMED OUT.")
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
            coloring=best_coloring
        )

def printMaster(str):
    if rank==0:
        print(str)

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
    printMaster(f"MPI size = {size}")

    random.seed(10)

    if len(sys.argv) < 2:
        print("Usage: python seq.py <instance>")
        sys.exit(1)
        
    instance = sys.argv[1]

    printMaster(f"Starting at: {time.strftime('%H:%M:%S', time.localtime())}\n")
    
    time_limit = 10000

    printMaster(f"Solving {instance}...")
    solve_instance_parallel(instance, time_limit)
    comm.barrier() # Ensure all ranks finished

if __name__ == "__main__":
    main()
