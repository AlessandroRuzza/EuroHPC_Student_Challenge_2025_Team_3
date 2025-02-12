import random
import time
import heapq
from copy import deepcopy
from collections import defaultdict

from algorithms.maxclique_heuristics import *
from algorithms.coloring_heuristics import *
from algorithms.branching_strategies import *


from utils import *


class Graph:

    # Constructor
    def __init__(self, num_nodes, 
                 coloring_algorithm=DSatur(), 
                 clique_algorithm=DLS(),
                 branching_strategy=DefaultBranchingStrategy()):
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


##########################################################################################

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

class BranchAndBoundNode:
    def __init__(self, union_find, added_edges, lb, ub):
        self.union_find = union_find
        self.added_edges = set(added_edges)
        self.lb = lb
        self.ub = ub

    def __lt__(self, other):
        return self.ub < other.ub

##########################################################################################

# lb - lower bound
# ub - upper bound
def branch_and_bound(graph, time_limit=10000):
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
    return best_ub, bestColoring

##########################################################################################

def solve_instance(filename):
    graph = parse_col_file(filename)

    # Set up algorithms
    graph.set_coloring_algorithm(DSatur())
    graph.set_clique_algorithm(DLS())
    graph.set_branching_strategy(DegreeBranchingStrategy())

    best_ub, bestColoring = branch_and_bound(graph)
    print(f"Best UB = {best_ub}")
    print(f"Is valid? {graph.validate(bestColoring)}")

def main():
    solve_instance("instances/anna.col")


if __name__ == "__main__":
    main()


