from mpi4py import MPI
import time

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

from utilities.utils import parse_col_file, output_results, Log

from graph.base import *

from algorithms.maxclique_heuristics import *
from algorithms.coloring_heuristics import *
from algorithms.branching_strategies import *

from copy import deepcopy

# Solver info
solverName = "MPI_BacktrackingDSatur_DLS"
solverVersion = "v1.0.1"

# Debug flags
debugSlave = False
debugBounds = True
debugQueue = False

# MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nodesPerSlave = 5
coresPerSlave = 32

# Branch parameters
worsenTolerance = 1
minQueueLenForPruning = size*2 * nodesPerSlave

# Logging parameters
log_file_path = "./log"
log = Log(log_file_path)

def printDebugSlave(str):
    """
    Print debug messages for slave processes if debugSlave is True
    
    :param str: message to print
    :type str: str
    """
    if debugSlave: print(str)

def printDebugBounds(str):
    """
    Print debug messages for bounds if debugBounds is True
    
    :param str: message to print
    :type str: str
    """
    if debugBounds: print(str)
    
def printConditional(str, condition):
    """
    Print a message if a condition is met.
    
    :param str: message to print
    :type str: str
    :param condition: condition to check
    :type condition: bool
    """
    if condition: print(str)
    
def printMaster(str):
    """
    Print a message if called by the master process

    :param str: message to print
    :type str: str
    """
    if rank==0:
        print(str)


def branch_node(graph, node):
    """
    Computing the children nodes of a given node in the branch and bound tree with its upper and lower bounds

    :param graph: graph to solve
    :type graph: Graph
    :param node: node to branch on
    :type node: BranchAndBoundNode
    :return: list of children nodes, one with the same color for the pair of nodes and one with different colors
    :rtype: list[BranchAndBoundNode]
    """

    # Find a pair of nodes to branch on
    u, v = graph.find_pair(node.union_find, node.added_edges)
    if u is None:
        return None
    
    childNodes = []

    # Branch 1: Same color
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

        clique = graph.find_max_clique(uf1, edges1)
        lb1 = len(clique)
        

        coloring = graph.find_coloring(uf1, edges1)
        ub1 = len(set(coloring))

        indep_num = coloring_to_independence_number(coloring)
        indep_LB = len(graph) / indep_num
        print(f"1 COL Indep. number LB = {indep_LB} ; Clique LB = {lb1}")

        COMPL_indep_num = len(graph.complement().find_max_clique(UnionFind(len(graph)), list()))
        print(f"COMPL Indep. number LB = {COMPL_indep_num} ; Clique LB = {lb1}")


        childNodes.append(BranchAndBoundNode(uf1, edges1, lb1, ub1))
        log.append(f"Node {node.id} branched by imposing {u}, {v} have the same color \n")
        log.append(f"Branch 1 child node results: \n")
        log.append(f"Clique (LB = {lb1}) = {clique}\n")
        log.append(f"Indep. LB = {indep_LB}\n")
        log.append(f"Coloring (UB = {ub1}) = {coloring}\n\n")

    # Branch 2: Different color
    uf2 = deepcopy(node.union_find)
    edges2 = deepcopy(node.added_edges)
    ru = uf2.find(u)
    rv = uf2.find(v)
    edges2.add((ru, rv))

    clique = graph.find_max_clique(uf2, edges2)
    lb2 = len(clique)
    coloring = graph.find_coloring(uf2, edges2)
    ub2 = len(set(coloring))

    indep_num = coloring_to_independence_number(coloring)
    indep_LB = len(graph) / indep_num
    print(f"2 COL Indep. number LB = {indep_LB} ; Clique LB = {lb2}")
    COMPL_indep_num = len(graph.complement().find_max_clique(UnionFind(len(graph)), list()))
    print(f"COMPL Indep. number LB = {COMPL_indep_num} ; Clique LB = {lb2}")

    childNodes.append(BranchAndBoundNode(uf2, edges2, lb2, ub2))

    log.append(f"Node {node.id} branched by imposing vertices {u}, {v} have different colors\n")
    log.append(f"Branch 2 child node results: \n")
    log.append(f"Clique (LB = {lb2}) = {clique}\n")
    log.append(f"Indep. LB = {indep_LB}\n")
    log.append(f"Coloring (UB = {ub2}) = {coloring}\n\n")

    return childNodes

def handle_slave(graph, slaveRank, 
                 queueLock: Condition, queue: list[BranchAndBoundNode],
                 best_ub_lock: Condition, best_ub, best_lb, best_coloring: list,
                 optimalEvent: Event, timeoutEvent: Event, 
                 start_time, time_limit, shared_node_id):
    
    """
    Manages the slave process from the master process in a parallel branch and bound algorithm. 
    Distributing the work to the slaves, and collecting results

    :param graph: The graph to solve.
    :type graph: Graph
    :param slaveRank: The rank of the slave process.
    :type slaveRank: int
    :param queueLock: A condition variable for synchronizing access to the queue.
    :type queueLock: Condition
    :param queue: The queue of nodes to be processed.
    :type queue: list[BranchAndBoundNode]
    :param best_ub_lock: A condition variable for synchronizing access to the best upper bound.
    :type best_ub_lock: Condition
    :param best_ub: The best upper bound found so far. best_ub[0] is the best upper bound.
    :type best_ub: list[int]
    :param best_lb: The best lower bound found so far. best_lb[0] is the best lower bound.
    :type best_lb: list[int]
    :param best_coloring: The best coloring found so far.
    :type best_coloring: list
    :param optimalEvent: An event that is set when an optimal solution is found.
    :type optimalEvent: Event
    :param timeoutEvent: An event that is set when the time limit is reached.
    :type timeoutEvent: Event
    :param start_time: The starting time of the computation.
    :type start_time: float
    :param time_limit: The time limit for the computation in seconds.
    :type time_limit: int
    :param shared_node_id: Id of the last node extracted from queue. shared_node_id[0] is the id.
    :type shared_node_id: list[int]
    """

    flag = True
    
    # Continuously check for nodes to solve until timeout or optimal solution found
    while True:
        if timeoutEvent.is_set() or optimalEvent.is_set():
            break

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            timeoutEvent.set()
            break

        with queueLock: 
            # Temp debug to validate worsenTolerance pruning strategy
            if not int(elapsed)%10 and flag:
                printConditional(f"Len Queue = {len(queue)}", debugQueue)
                flag=False
            elif int(elapsed)%10: 
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
                node = queue.pop(0)
                node.id = shared_node_id[0]
                shared_node_id[0] += 1
                nodes.append(node)
        
        pruneNodes = []
        optimalFound = False
        
        # Initialize pruning and optimal flags
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
    """
    Master process that sends nodes to slave processes and receives the new nodes generated by them
    until an optimal solution is found or the time limit is reached

    :param graph: graph to solve
    :type graph: Graph
    :param queue: queue of nodes to solve
    :type queue: list[BranchAndBoundNode]
    :param best_ub: best upper bound found so far
    :type best_ub: int
    :param best_lb: best lower bound found so far
    :type best_lb: int
    :param best_coloring: best coloring found so far
    :type best_coloring: list
    :param start_time: starting time
    :type start_time: float
    :param time_limit: time limit in seconds
    :type time_limit: int
    :return: best upper bound, best lower bound, best coloring
    :rtype: int, int, list
    """

    # Run threads that interface with each slave (ranks [1, 2, 3, ..., size-1] )
    # Initialize shared variables
    slaveHandlers: list[Thread] = []
    queueLock = Condition()
    best_ub_lock = Lock()
    best_ub = [best_ub,]
    best_lb = [best_lb,]
    optimalEvent = Event()
    timeoutEvent = Event()
    shared_node_id = [0,]

    for slaveRank in range(1, size):
        slaveHandlers.append(Thread(daemon=True, target=handle_slave, 
                                    args=(graph,slaveRank, queueLock,queue, 
                                          best_ub_lock,best_ub,best_lb,best_coloring, 
                                          optimalEvent,timeoutEvent, 
                                          start_time,time_limit, shared_node_id)))

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
    """
    Process executed by the slave that receives nodes from master, computes their lower bounds and upper bounds, and sends back the results
    
    :param graph: graph to solve
    :type graph: Graph
    """
    while True:
        # Wait to receive work from master
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
    """
    Implementation of a parallel branch and bound algorithm for the graph coloring problem using MPI

    :param graph: graph to solve
    :type graph: Graph
    :param time_limit: time limit in seconds
    :type time_limit: int
    """
    
    if rank==0:

        # Initializations
        start_time = time.time()
        n = len(graph)
        initial_uf = UnionFind(n)
        initial_edges = set()

        # Find initial lower bound and upper bound
        initial_clique = graph.find_max_clique(initial_uf, initial_edges)
        lb = len(initial_clique)
        cliqueTime = time.time() - start_time
        initial_coloring = graph.find_coloring(initial_uf, initial_edges)
        ub = len(set(initial_coloring))
        colorTime = time.time() - (start_time+cliqueTime)
        
        indep_num = coloring_to_independence_number(initial_coloring)
        print(f"Indep. number LB = {len(graph) / indep_num}")

        # Shared best upper bound
        best_ub = ub
        best_lb = lb
        queue = []

        log.append(f"NOTE: the nodes in the log may be out of order. Refer to the IDs\n")
        log.append(f"      the ID indicates the order of extraction from the queue. \n\n")
        log.append(f"Root Node: \n")
        log.append(f"Clique (LB = {lb}) = {initial_clique}\n")
        log.append(f"Coloring (UB = {ub}) = {initial_coloring}\n\n")

        print(f"Starting (UB, LB) = ({ub}, {lb})")
        print(f"Coloring time = {int(colorTime/60)}m {colorTime%60:.3f}s")
        print(f"Clique time = {int(cliqueTime/60)}m {cliqueTime%60:.3f}s")
        queue.append(BranchAndBoundNode(initial_uf, initial_edges, lb, ub))
        return master_branch_and_bound(graph, queue, best_ub, best_lb, initial_coloring, start_time, time_limit)
    else:
        slave_branch_and_bound(graph)
        return None, None, None

def solve_instance_parallel(filename, time_limit):
    """
    Solve a single instance using a parallel branch and bound algorithm within the time limit

    :param filename: instance to solve
    :type filename: str
    :param timeLimit: time limit in seconds
    :type timeLimit: int
    """
    graph = parse_col_file(filename)

    # Set up heuristics
    graph.set_coloring_algorithm(Parallel_BacktrackingDSatur(time_limit=0.6, num_workers=coresPerSlave))
    graph.set_clique_algorithm(ParallelDLS(num_workers=coresPerSlave, lambda_max_steps=10, dls_instance=DLS()))
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
            solver_name=solverName,
            solver_version=solverVersion,
            num_workers=size,
            num_cores=coresPerSlave,
            wall_time=wall_time,
            time_limit=time_limit,
            graph=graph,
            coloring=best_coloring
        )

    return chromatic_number, maxCliqueSize, best_coloring

def main():
    """
    Main Function.
    """
    printMaster(f"MPI size = {size}")

    random.seed(10)

    if len(sys.argv) < 2:
        print("Usage: python seq.py <instance>")
        sys.exit(1)
        
    instance = sys.argv[1]

    printMaster(f"Starting at: {time.strftime('%H:%M:%S', time.localtime())}\n")
    
    time_limit = 10000

    printMaster(f"Solving {instance}...")
    chromatic_number, maxCliqueSize, best_coloring = solve_instance_parallel(instance, time_limit)

    comm.barrier() # Ensure all ranks finished

    if rank > 0:
        comm.send(log, 0)
    else: # rank 0
        for i in range(1, size):
            otherLog = comm.recv(source=i)
            log.append(otherLog.log)

        log.append(f"Final results: \n")
        log.append(f"Chromatic number = {chromatic_number} \n")
        log.append(f"Max Clique Size = {maxCliqueSize} \n")
        log.append(f"Is Optimal? {'YES' if chromatic_number == maxCliqueSize else 'UNKNOWN'} \n")
        log.append(f"Best Coloring Found = {best_coloring} \n")
        log.printToFile()

if __name__ == "__main__":
    main()
