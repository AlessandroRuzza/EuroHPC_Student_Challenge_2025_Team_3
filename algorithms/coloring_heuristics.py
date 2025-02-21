from abc import ABC, abstractmethod
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np


class ColoringHeuristic(ABC):
    """
    Abstract base class for coloring heuristics
    """
    def uf_to_coloring(self, graph, uf):
        """
        Returns a coloring of the graph as a list where index is node and value is color.
        The coloring is created according to information in the UnionFind parameter

        :param graph: Graph to color
        :type graph: Graph
        :param uf: Data Structure to keep track of vertex colors
        :type uf: UnionFind
        :return: List of colors
        :rtype: list[int]
        """
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
        """
        Returns a set of colors assigned to neighbors of the node parameter in the graph

        :param graph: Graph to color
        :type graph: Graph
        :param uf: Data Structure to keep track of vertex colors
        :type uf: UnionFind
        :param added_edges: Data Structure to impose different colors on vertices
        :type added_edges: list[set(int,int)]
        :param coloring: coloring of the graph (index is node, value is color)
        :type coloring: list[int]
        :param node: target node for neighbor color retrieval
        :type node: int
        :return: List of colors
        :rtype: list[int]
        """
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
    DSatur coloring heuristic.
    The ordering in which colors are assigned to nodes is determined dynamically on node saturation.
    Ties resolved by node degree.
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
    """
    Backtracking DSatur coloring algorithm.
    Uses the DSATUR heuristic with backtracking to find a valid coloring with the minimum number of colors.
    Node selection is performed by saturation, breaking ties by degree. 
    Some randomness is added to explore the solution space.
    Backtracking happens when there are no possible color assignments that do not use more colors than the current best upper bound
    """
    def __init__(self, time_limit):
        """
        Constructor.

        :param time_limit: Max execution time in seconds for this algorithm.
        :type time_limit: float
        """
        super().__init__()
        self.dsatur = DSatur()
        self.time_limit = time_limit

    def find_coloring(self, graph, uf, added_edges):
        local_obj = BacktrackingDSatur(self.time_limit)
        return local_obj.__dsatur_backtracking__(graph, uf, added_edges)
    
    def __dsatur_backtracking__(self, graph, uf, added_edges):
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
        start_time = time.time()
        self.best_coloring = self.dsatur.find_coloring(graph, uf, added_edges)
        # Slaves save the best_ub value in the graph object
        self.best_num_colors = min(graph.best_ub, len(set(self.best_coloring))) # Initial UB
        
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
                    self.best_num_colors = current_max_color
                    self.best_coloring = coloring[:]
                return

            # Select the next node dynamically based on saturation and degree
            uncolored_nodes = [node for node in range(graph.num_nodes) if coloring[node] == -1]
            saturation = {
                node: len({coloring[neighbor] for neighbor in graph.adj_list[node] if coloring[neighbor] != -1})
                for node in uncolored_nodes
            }

            max_saturation = max(saturation.values())
            candidates = [node for node in uncolored_nodes if saturation[node] == max_saturation]

            # Weigh candidates by node degree
            degrees = [graph.degree(node) for node in candidates]
            total_degree = sum(degrees)
            probabilities = [deg / total_degree for deg in degrees]
            best_node = random.choices(candidates, weights=probabilities, k=1)[0]

            # Determine available colors
            neighbor_colors = {coloring[neighbor] for neighbor in graph.adj_list[best_node] if coloring[neighbor] != -1}
            available_colors = [color for color in range(self.best_num_colors) if color not in neighbor_colors]

            for color in available_colors:
                coloring[best_node] = color
                backtrack(coloring)
                coloring[best_node] = -1

        coloring = self.uf_to_coloring(graph, uf)
        backtrack(coloring)
        return self.best_coloring

class Parallel_BacktrackingDSatur(BacktrackingDSatur):
    """
    Parallel version of Backtracking DSatur that runs multiple instances on different threads
    Each instance will try different assignments due to the randomness in the ordering of nodes.
    """
    def __init__(self, time_limit, num_workers):
        """
        Constructor.

        :param time_limit: Max execution time in seconds for this algorithm.
        :type time_limit: float
        :param num_workers: Number of parallel threads that will run BacktrackingDSatur
        :type num_workers: int
        """
        super().__init__(time_limit)
        self.num_workers = num_workers
        
    def find_coloring(self, graph, uf, added_edges):
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            results = pool.map(super().find_coloring,
                            [graph] * self.num_workers,
                            [uf] * self.num_workers, 
                            [added_edges] * self.num_workers)
            
        # Return the best coloring found
        return min(results, key=lambda c: len(set(c)))
        

class TabuSearch(ColoringHeuristic):
    """
    Base Tabu Search heuristic for graph coloring.
    """

    def __init__(self, max_steps, tabu_size, initial_coloring_heuristic=DSatur()):
        """
        Constructor for the TabuSearch class.

        :param max_steps: Max number of steps for this algorithm.
        :type max_steps: int
        :param tabu_size: Size of the tabu list to avoid cycling back to recent solutions.
        :type tabu_size: int
        :param initial_coloring_heuristic: Initial coloring heuristic to use
        :type initial_coloring_heuristic: ColoringHeuristic
        """

        self.max_steps = max_steps
        self.tabu_size = tabu_size
        self.initial_coloring_heuristic = initial_coloring_heuristic

    def find_coloring(self, graph, union_find, added_edges):
        """
        Executes the Tabu Search algorithm to find a proper coloring of the graph.

        :param graph: Graph to color
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to add to the graph
        :type added_edges: list
        :return: List of colors
        :rtype: list[int]
        """
        current_coloring = self.initial_solution(graph, union_find, added_edges)
        best_coloring = current_coloring[:]
        best_num_colors = len(set(best_coloring))

        tabu_list = []
        steps = 0
        
        while steps < self.max_steps:
            neighborhood = self.get_neighborhood(current_coloring, graph, union_find, added_edges)
            if not neighborhood:
                break
            
            next_coloring = self.select_best_neighbor(neighborhood, tabu_list, best_num_colors)
            current_coloring = next_coloring[:]
            current_num_colors = len(set(current_coloring))

            if current_num_colors < best_num_colors:
                best_coloring = current_coloring[:]
                best_num_colors = current_num_colors

            tabu_list.append(current_coloring)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)

            steps += 1

        return best_coloring

    def initial_solution(self, graph, union_find, added_edges):
        """
        Generates an initial coloring solution using the specified initial coloring heuristic.

        :param graph: Graph to color
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to add to the graph
        :type added_edges: list
        :return: Initial coloring of the graph
        :rtype: list[int]
        """
        return self.initial_coloring_heuristic.find_coloring(graph, union_find, added_edges)

    def get_neighborhood(self, coloring, graph, union_find, added_edges):
        """
        Generates a neighborhood of solutions by modifying the current coloring.

        :param coloring: Current coloring of the graph
        :type coloring: list[int]
        :param graph: Graph to color
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to add to the graph
        :type added_edges: list
        :return: List of neighboring colorings
        :rtype: list[list[int]]
        """
        neighborhood = []
        for node in range(graph.num_nodes):
            current_color = coloring[node]
            for new_color in range(max(coloring) + 1):
                if new_color != current_color:
                    new_coloring = coloring[:]
                    new_coloring[node] = new_color
                    # Update the color of all nodes in the same union
                    for i in range(graph.num_nodes):
                        if union_find.find(i) == union_find.find(node):
                            new_coloring[i] = new_color
                    
                    if self.is_valid_coloring(graph, new_coloring, added_edges):
                        neighborhood.append(new_coloring)
        return neighborhood

    def select_best_neighbor(self, neighborhood, tabu_list, best_num_colors):
        """
        Selects the best neighbor from the neighborhood that is not in the tabu list.

        :param neighborhood: List of neighboring colorings
        :type neighborhood: list[list[int]]
        :param tabu_list: List of recently visited solutions
        :type tabu_list: list[list[int]]
        :param best_num_colors: Current best number of colors used
        :type best_num_colors: int
        :return: Best neighbor coloring
        :rtype: list[int]
        """
        best_neighbor = None
        best_neighbor_num_colors = float('inf')

        for neighbor in neighborhood:
            if neighbor not in tabu_list:
                num_colors = len(set(neighbor))
                if num_colors < best_neighbor_num_colors:
                    best_neighbor = neighbor
                    best_neighbor_num_colors = num_colors

        return best_neighbor if best_neighbor else random.choice(neighborhood)

    def is_valid_coloring(self, graph, coloring, added_edges):
        """
        Validates if the given coloring is valid according to the graph and added edges.

        :param graph: Graph to validate against
        :type graph: Graph
        :param coloring: Color assignment (index is node, value is color)
        :type coloring: list[int]
        :param added_edges: List of edges to add to the graph
        :type added_edges: list
        :return: True if the coloring is valid, False otherwise
        :rtype: bool
        """
        for a, b in added_edges:
            if coloring[a] == coloring[b]:
                return False
        try: 
            return graph.validate_coloring(coloring)
        except ValueError:
            return False

