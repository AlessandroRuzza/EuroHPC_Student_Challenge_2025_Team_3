
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
algorithms_dir = current_dir.parent / 'algorithms'
sys.path.append(str(algorithms_dir.parent))

from algorithms.maxclique_heuristics import *
from algorithms.coloring_heuristics import *
from algorithms.branching_strategies import *


class Graph:

    # Constructor
    def __init__(self, num_nodes, 
                 coloring_algorithm=DSatur(), 
                 clique_algorithm=DLS(),
                 branching_strategy=SaturationBranchingStrategy()):
        self.num_nodes = num_nodes
        self.adj_list = defaultdict(list)
        self.coloring_algorithm = coloring_algorithm
        self.clique_algorithm = clique_algorithm
        self.branching_strategy = branching_strategy


    ##### Graph operations

    def add_edge(self, u, v):
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def is_connected(self, u, v):
        return v in self.adj_list[u]

    def degree(self, node):
        return len(self.adj_list[node])


    ##### Coloring and clique algorithms

    def set_coloring_algorithm(self, algorithm):
        self.coloring_algorithm = algorithm

    def set_clique_algorithm(self, algorithm):
        self.clique_algorithm = algorithm

    def set_branching_strategy(self, strategy):
        self.branching_strategy = strategy

    def find_coloring(self, union_find, added_edges):
        if self.coloring_algorithm is None:
            raise ValueError("No coloring algorithm set")

        return self.coloring_algorithm.find_coloring(self, union_find, added_edges)

    def find_max_clique(self, union_find, added_edges):
        if self.clique_algorithm is None:
            raise ValueError("No clique algorithm set")

        return self.clique_algorithm.find_max_clique(self, union_find, added_edges)

    def find_pair(self, union_find, added_edges):
        if self.branching_strategy is None:
            raise ValueError("No branching strategy set")

        return self.branching_strategy.find_pair(self, union_find, added_edges)


    ##### Helper functions


    def __len__(self):
        return self.num_nodes

    def __str__(self):
        return "\n".join(f"{node}: {neighbors}" for node, neighbors in self.adj_list.items())

    ##### Coloring validation

    def validate(self, coloring):
        for node in range(self.num_nodes):
            color = coloring[node]
            neighbors = list(i for i in self.adj_list[node])
            neighborColors = list(coloring[i] for i in self.adj_list[node])

            if color in neighborColors:
                bad = next(n for n in range(len(neighborColors)) if neighborColors[n] == color)
                return f"FALSE. Incorrect coloring ({node+1}={color},{neighbors[bad]+1}={coloring[bad]})"
        return "TRUE."

    def correct_coloring_check(self, node, color, coloring):
        neighborColors = (coloring[i] for i in self.adj_list[node])
        return not color in neighborColors

