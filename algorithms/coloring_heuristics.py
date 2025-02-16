from abc import ABC, abstractmethod
from collections import defaultdict
import time

class ColoringHeuristic(ABC):
    """
    Abstract base class for coloring heuristics
    """
    def uf_to_coloring(self, graph, uf):
        coloring = [-1] * len(graph)

        # Normalise color ids in unionFind (by calling find on merged nodes)
        # This results in the roots of those unions not being colored
        for x in range(len(graph)):
            if uf.find(x) == x:
                coloring[x] = -1
            else:
                coloring[x] = uf.find(x)

        # Color the roots
        # The only nodes with -1 will be those that were never merged in unionFind
        for x in range(len(graph)):
            if coloring[x] != -1:
                coloring[uf.find(x)] = uf.find(x)

        # Normalize color ids to 0-k
        uniqueColors = [c for c in set(coloring) if c != -1]
        newColorIDs = {uniqueColors[i]: i for i in range(len(uniqueColors)) }
        newColorIDs[-1] = -1
        coloring = [newColorIDs[oldColor] for oldColor in coloring]

        return coloring

    def getNeighborColors(self, graph, uf, added_edges, coloring, node):
        # Determine available colors
        neighbor_colors = list(coloring[neighbor] for neighbor in graph.adj_list[node] if coloring[neighbor] >= 0)
        # Add neighbors according to added_edges
        neighbor_colors.append(coloring[n] for n,b in added_edges if uf.find(b) == uf.find(node) if coloring[n] >= 0)
        neighbor_colors.append(coloring[n] for a,n in added_edges if uf.find(a) == uf.find(node) if coloring[n] >= 0)
        neighbor_colors = set(neighbor_colors) # remove duplicates
        return neighbor_colors
    
    @abstractmethod
    def find_coloring(self, graph, union_find, added_edges):
        """
        Returns a coloring of the graph as a list where index is node and value is color

        :param graph: Graph to color
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to add to the graph
        :type added_edges: list
        :return: List of colors
        :rtype: list
        """
        pass

class DSatur(ColoringHeuristic):
    """
    DSatur coloring heuristic
    """

    def find_coloring(self, graph, uf, added_edges):
        coloring = self.uf_to_coloring(graph, uf)

        saturation = [0] * len(graph)
        for node in range(len(graph)):
            neighborColors = set(coloring[n] for n in graph.adj_list[node] if coloring[n] != -1)
            saturation[node] = len(neighborColors)

        uncolored_nodes = set(n for n in range(len(graph)) if coloring[n] == -1)

        while uncolored_nodes:
            # Select node with highest saturation, breaking ties by degree
            best_node = max(uncolored_nodes, key=lambda n: (saturation[n], graph.degree(n)))
            uncolored_nodes.remove(best_node)

            # Determine available colors
            neighbor_colors = self.getNeighborColors(graph, uf, added_edges, coloring, best_node)
            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[best_node] = color

            # Update saturation of neighbors
            for neighbor in graph.adj_list[best_node]:
                if coloring[neighbor] == -1:
                    saturation[neighbor] += 1

        return coloring

class BacktrackingDSatur(ColoringHeuristic):
    def __init__(self, time_limit):
        super().__init__()
        self.dsatur = DSatur()
        self.time_limit = time_limit

    def find_coloring(self, graph, uf, added_edges):
        local_obj = BacktrackingDSatur(self.time_limit)
        return local_obj.__dsatur_backtracking__(graph, uf, added_edges)
    
    def __dsatur_backtracking__(self, graph, uf, added_edges):
        start_time = time.time()
        self.best_coloring = self.dsatur.find_coloring(graph, uf, added_edges)
        # Slaves save the best_ub value in the graph object
        self.best_num_colors = min(graph.best_ub, len(set(self.best_coloring))) # Initial UB
        
        # msg = f"Backtrack. graphUB={graph.best_ub} ; init_Len={len(set(self.best_coloring))}"
        
        def backtrack(coloring):
            if time.time() - start_time > self.time_limit:
                return
            # Check added_edges constraint
            for a, b in added_edges:
                if coloring[a] == coloring[b]:
                    return
            if all(color != -1 for color in coloring):
                current_max_color = max(coloring) + 1
                if current_max_color < self.best_num_colors:
                    # validTxt = "Valid" if graph.validate(coloring) else "Invalid"
                    # print(f"Backtrack improved from {self.best_num_colors} to {current_max_color}. {validTxt}")
                    self.best_num_colors = current_max_color
                    self.best_coloring = coloring[:]
                return

            # Select the next node dynamically based on saturation and degree
            uncolored_nodes = [node for node in range(graph.num_nodes) if coloring[node] == -1]
            saturation = {node: len({coloring[neighbor] for neighbor in graph.adj_list[node] if coloring[neighbor] != -1}) for node in uncolored_nodes}
            best_node = max(uncolored_nodes, key=lambda n: (saturation[n], graph.degree(n)))

            # Determine available colors
            neighbor_colors = {coloring[neighbor] for neighbor in graph.adj_list[best_node] if coloring[neighbor] != -1}
            available_colors = [color for color in range(self.best_num_colors) if color not in neighbor_colors]

            for color in available_colors:
                coloring[best_node] = color
                backtrack(coloring)
                coloring[best_node] = -1

        coloring = self.uf_to_coloring(graph, uf)
        backtrack(coloring)
        # print(msg, f"; final={self.best_num_colors} ; final_Len={len(set(self.best_coloring))}")
        return self.best_coloring
