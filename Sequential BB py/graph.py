import time
import heapq
from copy import deepcopy
from collections import defaultdict
from os import listdir, stat
from os.path import isfile, join

from algorithms.maxclique_heuristics import *
from algorithms.coloring_heuristics import *
from algorithms.branching_strategies import *


from utils import *


class Graph:
    """
    Graph class
    """
    # Constructor
    def __init__(self, num_nodes, 
                 coloring_algorithm=DSatur(), 
                 clique_algorithm=DLS(),
                 branching_strategy=DefaultBranchingStrategy()):
        """
        Constructor for the Graph class

        :param num_nodes: Number of nodes in the graph
        :type num_nodes: int
        :param coloring_algorithm: Coloring algorithm to use
        :type coloring_algorithm: ColoringHeuristic
        :param clique_algorithm: Clique algorithm to use
        :type clique_algorithm: MaxCliqueHeuristic
        :param branching_strategy: Branching strategy to use
        :type branching_strategy: BranchingStrategy
        """
        self.num_nodes = num_nodes
        self.adj_list = defaultdict(list)
        self.coloring_algorithm = coloring_algorithm
        self.clique_algorithm = clique_algorithm
        self.branching_strategy = branching_strategy


    ##### Graph operations

    def add_edge(self, u, v):
        """
        Add an edge to the graph

        :param u: First node
        :type u: int
        :param v: Second node
        :type v: int
        :return: None
        :rtype: None
        """
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def is_connected(self, u, v):
        """
        Check if two nodes are connected

        :param u: First node
        :type u: int
        :param v: Second node
        :type v: int
        :return: True if the nodes are connected, False otherwise
        :rtype: bool
        """
        return v in self.adj_list[u]

    def degree(self, node):
        """
        Get the degree of a node

        :param node: Node to get the degree of
        :type node: int
        :return: Degree of the node
        :rtype: int
        """
        return len(self.adj_list[node])


    ##### Coloring and clique algorithms

    def set_coloring_algorithm(self, algorithm):
        """
        Method to set the coloring algorithm

        :param algorithm: Coloring algorithm to set
        :type algorithm: ColoringHeuristic
        """
        self.coloring_algorithm = algorithm

    def set_clique_algorithm(self, algorithm):
        """
        Method to set the clique algorithm

        :param algorithm: Clique algorithm to set
        :type algorithm: MaxCliqueHeuristic
        """
        self.clique_algorithm = algorithm

    def set_branching_strategy(self, strategy):
        """
        Method to set the branching strategy

        :param strategy: Branching strategy to set
        :type strategy: BranchingStrategy
        """
        self.branching_strategy = strategy

    def find_coloring(self, union_find, added_edges):
        """
        Main method to find a coloring of the graph

        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
        :raises ValueError: If no coloring algorithm is set
        :return: Coloring of the graph as a list of colors
        :rtype: list
        """
        if self.coloring_algorithm is None:
            raise ValueError("No coloring algorithm set")

        return self.coloring_algorithm.find_coloring(self, union_find, added_edges)

    def find_max_clique(self, union_find, added_edges):
        """
        Main method to find a maximum clique in the graph

        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
        :raises ValueError: If no clique algorithm is set
        :return: Maximum clique of the graph as a set of nodes
        :rtype: set
        """
        if self.clique_algorithm is None:
            raise ValueError("No clique algorithm set")

        return self.clique_algorithm.find_max_clique(self, union_find, added_edges)

    def find_pair(self, union_find, added_edges):
        """
        Main method to find a pair of nodes to branch on

        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
        :raises ValueError: If no branching strategy is set
        :return: Pair of nodes to branch on
        :rtype: tuple
        """
        if self.branching_strategy is None:
            raise ValueError("No branching strategy set")

        return self.branching_strategy.find_pair(self, union_find, added_edges)


    ##### Helper functions


    def __len__(self):
        """
        Get the number of nodes in the graph

        :return: Number of nodes in the graph
        :rtype: int
        """
        return self.num_nodes

    def __str__(self):
        """
        String representation of the graph

        :return: String representation of the graph
        :rtype: str
        """
        return "\n".join(f"{node}: {neighbors}" for node, neighbors in self.adj_list.items())

    ##### Coloring validation

    def validate(self, coloring):
        """
        Validate the coloring of the graph, checking if any node has a neighbor with the same color

        :param coloring: Coloring of the graph
        :type coloring: list
        :return: String indicating if the coloring is valid or not. If invalid, it returns the first invalid coloring found.
        :rtype: str
        """
        for node in range(self.num_nodes):
            color = coloring[node]
            neighbors = list(i for i in self.adj_list[node])
            neighborColors = list(coloring[i] for i in self.adj_list[node])

            if color in neighborColors:
                bad = next(n for n in range(len(neighborColors)) if neighborColors[n] == color)
                return f"FALSE. Incorrect coloring ({node+1}={color},{neighbors[bad]+1}={coloring[bad]})"
        return "TRUE."

    def correct_coloring_check(self, node, color, coloring):
        """
        Check if the coloring is correct for a given node

        :param node: Node to check
        :type node: int
        :param color: Color of the node
        :type color: int
        :param coloring: Coloring of the graph
        :type coloring: list
        :return: True if the coloring is correct, False otherwise
        :rtype: bool
        """
        neighborColors = (coloring[i] for i in self.adj_list[node])
        return not color in neighborColors


##########################################################################################

class UnionFind:
    """
    Data structure to keep track of vertex colors
    """
    def __init__(self, size):
        """
        Constructor for the UnionFind class

        :param size: Size of the UnionFind data structure
        :type size: int
        """
        self.parent = list(range(size))

    def find(self, x):
        """
        Find the color of the node

        :param x: Node to find the color of
        :type x: int
        :return: Color of the node
        :rtype: int
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        Make the colors of two nodes the same

        :param x: First node
        :type x: int
        :param y: Second node
        :type y: int
        """
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

class BranchAndBoundNode:
    """
    Node for the branch and bound algorithm
    """
    def __init__(self, union_find, added_edges, lb, ub):
        """
        Constructor for the BranchAndBoundNode class

        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: set
        :param lb: Lower bound of the node
        :type lb: int
        :param ub: Upper bound of the node
        :type ub: int
        """
        self.union_find = union_find
        self.added_edges = set(added_edges)
        self.lb = lb
        self.ub = ub

    def __lt__(self, other):
        """
        Compare two nodes based on their upper bound

        :param other: Other node to compare to
        :type other: BranchAndBoundNode
        :return: True if the current node has a smaller upper bound, False otherwise
        :rtype: bool
        """
        return self.ub < other.ub

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

    while queue:

        current_ub, node = heapq.heappop(queue)
        current_lb = node.lb
        if node.lb >= best_ub:
            continue
        if current_ub < best_ub:
            print(f"IMPROVED UB! {current_ub}")
            best_ub = current_ub
        if current_lb > best_lb:
            print(f"IMPROVED LB! {current_lb}")
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
    return best_ub, bestColoring, is_over_time_limit

##########################################################################################

def solve_instance(filename, timeLimit):
    graph = parse_col_file(filename)

    # Configuration parameters
    solver_name = "sequential_DSatur_DLS_DegreeBranching"
    solver_version = "v1.0.1"

    # Set up algorithms
    graph.set_coloring_algorithm(DSatur())
    graph.set_clique_algorithm(DLS())
    graph.set_branching_strategy(DegreeBranchingStrategy())

    # MPI parameters
    num_workers = 5
    num_cores = 6


    start_time = time.time()
    best_ub, bestColoring, isOverTimeLimit = branch_and_bound(graph, timeLimit)
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
    print(f"Best UB = {best_ub}")
    print(f"Is valid? {isValid}")
    print(f"Passed time limit? {isOverTimeLimit}")

    return isValid and not isOverTimeLimit

def main():
    instance_files = [join("./instances/", f) for f in listdir("./instances/") if isfile(join("./instances/", f))]
    # Sort by file size (bigger graphs take more time)
    instance_files = sorted(instance_files, key=lambda f: (stat(f).st_size))

    badInstances = ("myciel")
    for bad in badInstances:
        instance_files = [f for f in instance_files if not f.startswith(f"./instances/{bad}")]

    # longest instances: 
    #   fpsol2  (27.7s)
    #   inithx  (>10k s)
    #   all "myciel*" instances solved to optimality, but the lb is never improved (due to maxClique = 2, chromatic number > 2)
    #       so they keep running until the time limit (or all possible combinations are assigned in nodes)
    #   

    #  To clear instances solved
    # os.remove("solved_instances.txt") 

    start_from_idx = 0
    delay = 2
    timeLimit = 10000

    instance_files = instance_files[start_from_idx::]
    i = start_from_idx+1

    with open("solved_instances.txt", "w") as out:
        for instance in instance_files:
            print(f"Solving {instance}... #{i}/{len(instance_files)}")
            start = time.time()
            optimal = solve_instance(instance, timeLimit)
            elapsed = time.time() - start
            print(f"Time elapsed: {elapsed}s")

            print(f"Waiting {delay} secs to show result.")
            time.sleep(delay)
            print()
            i+=1

            if optimal:
                out.write(instance + "\n")


if __name__ == "__main__":
    main()


