from abc import ABC, abstractmethod
from collections import defaultdict

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
    def __init__(self):
        super().__init__()
        self.DSatur = DSatur()

    def find_coloring(self, graph, union_find, added_edges):
        init_coloring = self.uf_to_coloring(graph, union_find)

        saturation = [0] * len(graph)
        for node in range(len(graph)):
            neighborColors = set(init_coloring[n] for n in graph.adj_list[node] if init_coloring[n] != -1)
            saturation[node] = len(neighborColors)      

        uncolored_nodes = list(n for n in range(len(graph)) if init_coloring[n] == -1)
        uncolored_nodes = sorted(uncolored_nodes, key=lambda n: (saturation[n], graph.degree(n)))
        assignments = [init_coloring[i] for i in range(len(graph))]

        # Apply DSatur to find an initial UB
        # A k-coloring of k<=ub exists
        best_coloring = self.DSatur.find_coloring(graph, union_find, added_edges)
        ub = len(set(best_coloring))
        
        while True:
            # Search for k-coloring k<ub
            i = 0
            while i < len(uncolored_nodes):
                if i<0: # no possible assignments
                    break

                # Pick node
                curr_node = uncolored_nodes[i]

                # Assign color
                neighborColors = self.getNeighborColors(graph, union_find, added_edges, assignments, curr_node)
                new_color = assignments[curr_node]+1
                while new_color in neighborColors:
                    new_color += 1

                if new_color < ub-1: # min improvement uses 1 less color 
                                     # (e.g 2-coloring uses colors 0,1 -> max color for 1-coloring is 0 < 2-1)
                    # Assign color, forward step
                    assignments[curr_node] = new_color
                    i += 1 
                else:
                    # Do backward step
                    assignments[curr_node] = -1
                    i -= 1 
                    continue

            # If found, reduce ub
            if i>=0:
                ub = len(set(assignments))
                best_coloring = assignments.copy()
            else:
                break
        
        # When new assignment was not found
        return best_coloring


