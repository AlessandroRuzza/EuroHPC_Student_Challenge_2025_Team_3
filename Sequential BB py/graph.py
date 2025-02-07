import random
import time
import heapq
from copy import deepcopy
from collections import defaultdict

class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_list = defaultdict(list)

    def add_edge(self, u, v):
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def is_connected(self, u, v):
        return v in self.adj_list[u]

    def degree(self, node):
        return len(self.adj_list[node])

    def dsatur(self, uf, added_edges):
        coloring = [-1] * self.num_nodes

        # Normalise color ids in unionFind (by calling find on merged nodes)
        # This results in the roots of those unions not being colored
        for x in range(self.num_nodes):
            if(uf.find(x) == x):
                coloring[x] = -1
            else:
                coloring[x] = uf.find(x)

        # Color the roots
        # The only nodes with -1 will be those that were never merged in unionFind
        for x in range(self.num_nodes):
            if coloring[x] != -1:
                coloring[uf.find(x)] = uf.find(x)

        saturation = [0] * self.num_nodes
        for node in range(self.num_nodes):
            neighborColors = set(coloring[n] for n in self.adj_list[node] if coloring[n] != -1)
            saturation[node] = len(neighborColors)

        uncolored_nodes = set(n for n in range(self.num_nodes) if coloring[n] == -1)

        # print(uf.parent)
        # print(uncolored_nodes)
        while uncolored_nodes:
            # Select node with highest saturation, breaking ties by degree
            best_node = max(uncolored_nodes, key=lambda n: (saturation[n], self.degree(n)))
            uncolored_nodes.remove(best_node)

            # Determine available colors
            neighbor_colors = set(coloring[neighbor] for neighbor in self.adj_list[best_node] if coloring[neighbor] >= 0)
            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[best_node] = color

            # Update saturation of neighbors
            for neighbor in self.adj_list[best_node]:
                if coloring[neighbor] == -1:
                    saturation[neighbor] += 1

        return coloring

    def heuristic_max_clique(self, union_find, added_edges):
    # Simplified greedy max clique heuristic
        merged = {}
        
        for u in range(len(self)):
            root = union_find.find(u)
            if root not in merged:
                merged[root] = set()
            for v in self.adj_list[u]:
                merged[root].add(union_find.find(v))
            for a, b in added_edges:
                ra = union_find.find(a)
                rb = union_find.find(b)
                if ra == root:
                    merged[root].add(rb)
                if rb == root:
                    merged[root].add(ra)
        
        clique = []
        candidates = list(merged.keys())
        
        while candidates:
            node = max(candidates, key=lambda x: len(merged[x]))
            clique.append(node)
            candidates = [n for n in candidates if n in merged[node]]
        return len(clique)

    def greedy_max_clique(self):
        remaining_nodes = list(range(self.num_nodes))
        clique = []

        while remaining_nodes:
            # Select node with max degree
            selected_node = max(remaining_nodes, key=self.degree)
            clique.append(selected_node)

            # Keep only neighbors of the selected node
            remaining_nodes = [node for node in remaining_nodes if node != selected_node and self.is_connected(selected_node, node)]

        return len(clique)

    def __len__(self):
        return self.num_nodes

    def __str__(self):
        return "\n".join(f"{node}: {neighbors}" for node, neighbors in self.adj_list.items())

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


def generate_random_graph(num_nodes, density):
    graph = Graph(num_nodes)
    random.seed(time.time())

    for i in range(1, num_nodes):
        for j in range(i):
            if random.random() < density:
                graph.add_edge(i, j)

    return graph


def parse_col_file(file_path):
    graph = None

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
                    graph.add_edge(node1, node2)
                continue

    return graph

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

def branch_and_bound(graph, time_limit=10000):
    n = len(graph)
    initial_uf = UnionFind(n)
    initial_edges = set()

    lb = graph.heuristic_max_clique(initial_uf, initial_edges)
    # lb = graph.greedy_max_clique()
    ub = len(set(graph.dsatur(initial_uf, initial_edges)))

    best_ub = ub
    best_lb = lb
    queue = []
    heapq.heappush(queue, (ub, BranchAndBoundNode(initial_uf, initial_edges, lb, ub)))

    print(f"Initial (UB,LB) = ({ub},{lb})")
    input("Press to continue: ")

    current_ub, node = None, None
    current_lb = None

    while queue:

        # print(f"Remaining: {len(queue)}")

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

        # Non-adjacent pair
        found = False
        for u in range(n):
            for v in range(u+1, n):
                ru = node.union_find.find(u)
                rv = node.union_find.find(v)
                if ru == rv:
                    continue
                adj = (v in graph.adj_list[u]) or ((ru, rv) in node.added_edges) or ((rv, ru) in node.added_edges)
                if not adj:
                    found = True
                    break
            if found:
                break

        if not found:
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

            lb1 = graph.heuristic_max_clique(uf1, edges1)
            # lb1 = graph.greedy_max_clique()
            ub1 = len(set(graph.dsatur(uf1, edges1)))
            if lb1 < best_ub:
                heapq.heappush(queue, (ub1, BranchAndBoundNode(uf1, edges1, lb1, ub1)))

        # Branch 2: different color
        uf2 = deepcopy(node.union_find)
        edges2 = deepcopy(node.added_edges)
        ru = uf2.find(u)
        rv = uf2.find(v)
        if (ru, rv) not in edges2 and (rv, ru) not in edges2:
            edges2.add((ru, rv))

        lb2 = graph.heuristic_max_clique(uf2, edges2)
        # lb2 = graph.greedy_max_clique()
        ub2 = len(set(graph.dsatur(uf2, edges2)))
        if lb2 < best_ub:
            heapq.heappush(queue, (ub2, BranchAndBoundNode(uf2, edges2, lb2, ub2)))
        
    bestColoring = graph.dsatur(node.union_find, node.added_edges)
    return best_ub, bestColoring

def main():
    graph = parse_col_file("anna.col")

    print("Coloring graph using DSATUR...")
    coloring = graph.dsatur(UnionFind(graph.num_nodes), set())
    print(f"Coloring result: {coloring}")
    print(f"Colors used: {max(coloring)+1}")

    print("Finding max clique using greedy method...")
    max_clique_size = graph.greedy_max_clique()
    print(f"Max clique size: {max_clique_size}")

    print(f"Is valid? {graph.validate(coloring)}")

    best_ub, bestColoring = branch_and_bound(graph)
    print(f"Best UB = {best_ub}")
    print(f"Is valid? {graph.validate(bestColoring)}")


if __name__ == "__main__":
    main()



import heapq
from copy import deepcopy

# lb - lower bound
# ub - upper bound

