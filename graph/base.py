import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
algorithms_dir = current_dir.parent / 'algorithms'
sys.path.append(str(algorithms_dir.parent))

from algorithms.maxclique_heuristics import *
from algorithms.coloring_heuristics import *
from algorithms.branching_strategies import *

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
    Node for the branch and bound algorithm.
    Nodes have an id purely for logging purposes.
    """
    def __init__(self, union_find, added_edges, lb, ub):
        """
        Constructor for the BranchAndBoundNode class.

        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: set
        :param lb: Lower bound of the node
        :type lb: int
        :param ub: Upper bound of the node
        :type ub: int
        """
        self.id = -1
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

class Graph:
    """
    Class containing all information about the graph instance.
    """

    def __init__(self, num_nodes, 
                 coloring_algorithm=DSatur(), 
                 clique_algorithm=DLS(),
                 branching_strategy=SaturationBranchingStrategy()):
        """
        Constructor.

        :param num_nodes: Number of vertices in the graph
        :type num_nodes: int
        :param coloring_algorithm: Heuristic used to approximate the upper bound for the chromatic number
        :type coloring_algorithm: ColoringHeuristic
        :param clique_algorithm: Heuristic used to approximate the lower bound for the chromatic number
        :type clique_algorithm: MaxCliqueHeuristic
        :param branching_strategy: Heuristic used to branch nodes for this graph instance
        :type branching_strategy: BranchingStrategy
        """
        self.num_nodes = num_nodes
        self.adj_list = defaultdict(list)
        self.coloring_algorithm = coloring_algorithm
        self.clique_algorithm = clique_algorithm
        self.branching_strategy = branching_strategy
        self.best_ub = num_nodes


    ##### Graph operations

    def add_edge(self, u, v):
        """
        Add an edge between nodes u and v

        :param u: a node of the graph
        :type u: int
        :param v: a node of the graph
        :type v: int
        """
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def is_connected(self, u, v):
        """
        Returns true if an edge connecting u and v exists in this graph

        :param u: a node of the graph
        :type u: int
        :param v: a node of the graph
        :type v: int
        :return: true if an edge connecting u and v exists in this graph
        :rtype: bool
        """
        return v in self.adj_list[u]

    def degree(self, node):
        """
        Returns the degree of the node parameter 
        (min 1 as all nodes are connected to themselves)

        :param node: a node of the graph
        :type node: int
        :return: degree of the node parameter
        :rtype: int
        """
        return len(self.adj_list[node])+1 # +1 Because every node is adj to itself


    ##### Coloring and clique algorithms

    def set_coloring_algorithm(self, algorithm):
        """
        Set the coloring heuristic.

        :param coloring_algorithm: Heuristic used to approximate the upper bound for the chromatic number
        :type coloring_algorithm: ColoringHeuristic
        """
        self.coloring_algorithm = algorithm

    def set_clique_algorithm(self, algorithm):
        """
        Set the Max Clique heuristic.
        
        :param clique_algorithm: Heuristic used to approximate the lower bound for the chromatic number
        :type clique_algorithm: MaxCliqueHeuristic
        """
        self.clique_algorithm = algorithm

    def set_branching_strategy(self, strategy):
        """
        Set the Branching heuristic.

        :param branching_strategy: Heuristic used to branch nodes for this graph instance
        :type branching_strategy: BranchingStrategy
        """
        self.branching_strategy = strategy

    def find_coloring(self, union_find, added_edges):
        """
        Uses the graph's coloring heuristic to find a proper coloring of the graph, then returns it.
        
        :param graph: Graph to color
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to add to the graph
        :type added_edges: list
        :raise ValueError: if the coloring algorithm is not set
        :return: Coloring of the graph as a list where index is node and value is color
        :rtype: list
        """
        if self.coloring_algorithm is None:
            raise ValueError("No coloring algorithm set")

        return self.coloring_algorithm.find_coloring(self, union_find, added_edges)

    def find_max_clique(self, union_find, added_edges):
        """
        Uses the graph's Max Clique heuristic to find a set of nodes that form the maximum clique of the graph, then returns it.

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
        :raise ValueError: if the clique algorithm is not set
        :return: Set of nodes that form the maximum clique
        :rtype: set
        """
        if self.clique_algorithm is None:
            raise ValueError("No clique algorithm set")

        return self.clique_algorithm.find_max_clique(self, union_find, added_edges)

    def find_pair(self, union_find, added_edges):
        """
        Uses the graph's Branching heuristic to find a pair of nodes to branch on, then returns it.

        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
        :raise ValueError: if the branching strategy is not set
        :return: Pair of best nodes to branch on
        :rtype: set
        """
        if self.branching_strategy is None:
            raise ValueError("No branching strategy set")

        return self.branching_strategy.find_pair(self, union_find, added_edges)


    ##### Helper functions


    def __len__(self):
        """
        :return: Number of nodes in the graph
        :rtype: int
        """
        return self.num_nodes

    def __str__(self):
        """
        String format: <br>
        0: \<adj_list of node 0\>  <br>
        1: \<adj_list of node 1\>  <br>
        2: \<adj_list of node 2\>  <br>
        etc. for each node in the graph

        :return: Graph adjacency list as a string.
        :rtype: int
        """
        return "\n".join(f"{node}: {neighbors}" for node, neighbors in self.adj_list.items())

    ##### Validation

    def validate_max_clique(self, clique):
        """
        Validates that the given set of nodes forms a max clique.

        :param clique: set containing max clique
        :type clique: set[int]
        :raise ValueError: if the clique is not valid.
        :return: True if the clique is proper.
        :rtype: bool
        """

        for i in clique:
            for j in clique:
                if i == j:
                    continue
                if not self.is_connected(i, j):
                    raise ValueError(f"Incorrect clique found. The nodes {i} and {j} are not connected.")
        return True

                


    def validate_coloring(self, coloring):
        """
        Validates a coloring by checking the whole graph for conflicts.

        :param coloring: color assignment (index is node, value is color)
        :type coloring: list[int]
        :raise ValueError: if the coloring is not valid
        :return: True if the coloring is proper.
        :rtype: bool
        """
        for node in range(self.num_nodes):
            color = coloring[node]
            neighbors = list(i for i in self.adj_list[node])
            neighborColors = list(coloring[i] for i in self.adj_list[node])

            if color in neighborColors:
                bad = next(n for n in range(len(neighborColors)) if neighborColors[n] == color)
                raise ValueError(f"Incorrect coloring ({node+1}={color},{neighbors[bad]+1}={coloring[bad]})")
        return True

    def validate(self, coloring, clique):
        """
        Validates if the heuristics used return a valid coloring and clique

        :param coloring: color assignment (index is node, value is color)
        :type coloring: list[int]
        :param clique: set containing max clique
        :type clique: set[int]
        :raise ValueError: if one of the solutions are not valid
        :return: True if the solutions found are valid
        :rtype: bool
        """
        print("yooo")
        return self.validate_coloring(coloring) and self.validate_max_clique(clique)
