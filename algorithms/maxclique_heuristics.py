from abc import ABC, abstractmethod
from collections import defaultdict


import random

class MaxCliqueHeuristic(ABC):
    """
    Abstract base class for maximum clique heuristics
    """
    @abstractmethod
    def find_max_clique(self, graph, union_find, added_edges):
        """
        Returns a set of nodes that form the maximum clique found

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        :param union_find: Data structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
        :return: Set of nodes that form the maximum clique
        :rtype: set
        """
        pass



class GreedyCliqueFinder(MaxCliqueHeuristic):
    """
    First implementation ofGreedy algorithm to find the maximum clique in a graph
    """
    def find_max_clique(self, graph, union_find, added_edges):
        remaining_nodes = list(range(len(graph)))
        clique = []

        while remaining_nodes:
            selected_node = max(remaining_nodes, key=graph.degree)
            clique.append(selected_node)
            remaining_nodes = [node for node in remaining_nodes 
                             if node != selected_node and graph.is_connected(selected_node, node)]

        return clique



class HeuristicCliqueFinder(MaxCliqueHeuristic):
    """
    Implementation of Greedy algorithm using union-find data structure to find the maximum clique in a graph
    """
    def find_max_clique(self, graph, union_find, added_edges):
        merged = {}
        
        for u in range(len(graph)):
            root = union_find.find(u)
            if root not in merged:
                merged[root] = set()
            for v in graph.adj_list[u]:
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
        return clique

class BaseDLS(MaxCliqueHeuristic):
    """Base class for Dynamic Local Search implementations"""
    def __init__(self, max_steps=100, penalty_delay=1):
        """
        Constructor for the BaseDLS class

        :param max_steps: Number of iterations to run the algorithm before stopping
        :type max_steps: int
        :default max_steps: 100
        :param penalty_delay: Delay in penalty updates
        :type penalty_delay: int
        :default penalty_delay: 1
        """
        self.max_steps = max_steps
        self.penalty_delay = penalty_delay
        self.penalties = {}
        self.current_clique = None
        self.best_clique = None
        self.step_count = 0
    
    def initialize_search(self, graph):
        """Initialize the parameters before starting the search

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        """
        self.current_clique = {random.choice(range(graph.num_nodes))}
        self.best_clique = set(self.current_clique)
        if len(self.penalties) != graph.num_nodes:
            self.penalties = {v: 0 for v in range(graph.num_nodes)}
        self.step_count = 0

    def update_penalties(self):
        """Update the penalties for the vertices in the current clique

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        """
        for v in self.current_clique:
            self.penalties[v] += 1
        if self.step_count % self.penalty_delay == 0:
            for v in self.penalties:
                self.penalties[v] = max(0, self.penalties[v] - 1)

    def select_min_penalty(self, vertices):
        """Select the vertex with the minimum penalty from the candidates

        :param vertices: List of vertices to choose from
        :type vertices: list
        :return: Vertex with the minimum penalty
        """
        min_penalty = min(self.penalties[v] for v in vertices)
        return random.choice([v for v in vertices if self.penalties[v] == min_penalty])

    def expand(self, graph, clique):
        """Expand the current clique by adding suitable vertices.

        :param clique: Current clique
        :type clique: set
        :param penalties: Dictionary of penalties for each vertex
        :type penalties: dict
        :return: Expanded clique
        """
        while True:
            candidates = [v for v in range(graph.num_nodes) 
                        if v not in clique and 
                        all(graph.is_connected(v, u) for u in clique)]
            if not candidates:
                break
            clique.add(self.select_min_penalty(candidates))
        return clique

    def plateau_search(self, graph, clique):
        """Swap vertices to maintain clique size while exploring plateau.

        :param clique: Current clique
        :type clique: set
        :param penalties: Dictionary of penalties for each vertex
        :type penalties: dict
        """
        steps = 0
        while steps < self.max_steps:
            candidates = [v for v in range(graph.num_nodes) 
                        if v not in clique and 
                        sum(graph.is_connected(v, u) for u in clique) == len(clique) - 1]
            if not candidates:
                break
            v = self.select_min_penalty(candidates)
            u = next(u for u in clique if not graph.is_connected(v, u))
            clique.remove(u)
            clique.add(v)
            steps += 1
        return clique

    @abstractmethod
    def single_step(self, graph, union_find, added_edges):
        """Perform one step of the DLS algorithm.

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to be added to the graph
        :type added_edges: list
        :raise NotImplementedError: If the method is not implemented
        :return: The best clique found in the current step
        :rtype: set
        """
        raise NotImplementedError

    def find_max_clique(self, graph, union_find, added_edges):
        """Run the full DLS search

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to be added to the graph
        :type added_edges: list
        :return: The best clique found
        :rtype: set
        """
        self.initialize_search(graph)
        for _ in range(self.max_steps):
            self.single_step(graph, union_find, added_edges)

        return self.best_clique
    


class DLS(BaseDLS):
    """Normal DLS implementation"""

    def single_step(self, graph, union_find, added_edges):
        if self.current_clique is None:
            self.initialize_search(graph)

        self.expand(graph, self.current_clique)
        self.plateau_search(graph, self.current_clique)
        
        if len(self.current_clique) > len(self.best_clique):
            self.best_clique = set(self.current_clique)
        
        self.update_penalties()
        
        # Perturbation
        if self.penalty_delay > 1:
            self.current_clique = {max(self.current_clique, key=lambda v: self.penalties[v])}
        else:
            self.current_clique = {random.choice(range(graph.num_nodes))}
        
        self.step_count += 1
        return self.best_clique

class DLSwithColors(BaseDLS):
    """
    Dynamic Local Search (DLS) implementation that uses information about colors of vertices for finding maximum clique
    """
    def get_vertex_color(self, v, union_find):
        """Get the color of a vertex.

        :param v: Vertex to get the color of
        :type v: int
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :return: Color of the vertex
        :rtype: int
        """
        return union_find.find(v)
    
    def get_clique_colors(self, clique, union_find):
        """Get the set of colors present in the current clique.

        :param clique: Current clique
        :type clique: set
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :return: Set of colors present in the current clique
        :rtype: set
        """
        return {self.get_vertex_color(v, union_find) for v in clique}

    def expand(self, graph, clique, union_find):
        """Color-aware expansion of the current clique by adding vertices with different colors.

        :param clique: Current clique
        :type clique: set
        :param penalties: Dictionary of penalties for each vertex
        :type penalties: dict
        :return: Expanded clique
        """
        while True:
            clique_colors = self.get_clique_colors(clique, union_find)
            candidates = [
                v for v in range(graph.num_nodes) 
                if v not in clique and 
                all(graph.is_connected(v, u) for u in clique) and
                self.get_vertex_color(v, union_find) not in clique_colors
            ]
            if not candidates:
                break
            clique.add(self.select_min_penalty(candidates))
        return clique

    def plateau_search(self, graph, clique, union_find):
        """Color-aware plateau search by swapping vertices to maintain clique size with vertices of the same color or new colors.

        :param clique: Current clique
        :type clique: set
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        """
        steps = 0
        while steps < self.max_steps:
            clique_colors = self.get_clique_colors(clique, union_find)
            swap_candidates = []
            for u in clique:
                u_color = self.get_vertex_color(u, union_find)
                potential_swaps = [
                    (v, u) for v in range(graph.num_nodes)
                    if v not in clique and
                    sum(graph.is_connected(v, w) for w in clique) == len(clique) - 1 and
                    (self.get_vertex_color(v, union_find) == u_color or 
                     self.get_vertex_color(v, union_find) not in clique_colors)
                ]
                swap_candidates.extend(potential_swaps)
            
            if not swap_candidates:
                break
                
            v, u = min(swap_candidates, key=lambda x: self.penalties[x[0]])
            clique.remove(u)
            clique.add(v)
            steps += 1
        return clique

    def single_step(self, graph, union_find, added_edges):
        if self.current_clique is None:
            self.initialize_search(graph)

        self.expand(graph, self.current_clique, union_find)
        self.plateau_search(graph, self.current_clique, union_find)
        
        if len(self.current_clique) > len(self.best_clique):
            self.best_clique = set(self.current_clique)
        
        self.update_penalties()
        
        # Color-aware perturbation
        if self.penalty_delay > 1:
            self.current_clique = {max(self.current_clique, key=lambda v: self.penalties[v])}
        else:
            self.current_clique = {random.choice(range(graph.num_nodes))}
        
        self.step_count += 1
        return self.best_clique



class DLSAdaptive(BaseDLS):
    """
    Adaptive DLS implementation that switches between basic and color-aware strategies
    based on the state of the coloring and search progress.
    """
    def __init__(self, max_steps=100, initial_penalty_delay=1, color_threshold=0.75):
        """
        Constructor for the DLSAdaptive class

        :param max_steps: Number of iterations to run the algorithm before stopping
        :type max_steps: int
        :param initial_penalty_delay: Delay in penalty updates
        :type initial_penalty_delay: int
        :param color_threshold: Threshold for switching between strategies (values between 0 and 1)
        :type color_threshold: float
        """
        super().__init__(max_steps, initial_penalty_delay)
        self.color_threshold = color_threshold
        self.basic_dls = DLS(max_steps, initial_penalty_delay)
        self.color_dls = DLSwithColors(max_steps, initial_penalty_delay)
        self.current_strategy = None

    def single_step(self, graph, union_find, added_edges):
        if self.current_clique is None:
            self.initialize_search(graph)

        # Choose strategy based on coloring ratio
        colored_vertices = len(set(union_find.find(v) for v in range(graph.num_nodes)))
        color_ratio = colored_vertices / graph.num_nodes
        
        # Choose and sync strategy
        strategy = self.color_dls if color_ratio >= self.color_threshold else self.basic_dls
        strategy.current_clique = self.current_clique
        strategy.best_clique = self.best_clique
        strategy.penalties = self.penalties
        strategy.step_count = self.step_count

        # Perform single step with chosen strategy
        result = strategy.single_step(graph, union_find, added_edges)

        # Sync back the state
        self.current_clique = strategy.current_clique
        self.best_clique = strategy.best_clique 
        self.penalties = strategy.penalties
        self.step_count = strategy.step_count
        return self.best_clique


class DLSIncreasingPenalty(BaseDLS):
    """DLS implementation with increasing penalty delay
    After increase_interval steps without improvement, the penalty delay is increased by 1,
    saturating at max_penalty_delay
    """
    def __init__(self, max_steps=100, initial_penalty_delay=1, increase_interval=10, max_penalty_delay=5):
        """
        Constructor for the DLSIncreasingPenalty class

        :param max_steps: Number of iterations to run the algorithm before stopping
        :type max_steps: int
        :param initial_penalty_delay: Delay in penalty updates
        :type initial_penalty_delay: int
        :param increase_interval: Interval for increasing penalty delay
        :type increase_interval: int
        :param max_penalty_delay: Maximum penalty delay
        :type max_penalty_delay: int
        """
        super().__init__(max_steps, initial_penalty_delay)
        self.increase_interval = increase_interval
        self.max_penalty_delay = max_penalty_delay
        self.steps_without_improvement = 0

    def single_step(self, graph, union_find, added_edges):
        if self.current_clique is None:
            self.initialize_search(graph)

        self.expand(graph, self.current_clique)
        self.plateau_search(graph, self.current_clique)
        
        # Track improvements and adjust penalty delay
        if len(self.current_clique) > len(self.best_clique):
            self.best_clique = set(self.current_clique)
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
            new_penalty_delay = 1 + (self.steps_without_improvement // self.increase_interval)
            self.penalty_delay = min(new_penalty_delay, self.max_penalty_delay)
        
        self.update_penalties()
        
        # Perturbation
        if self.penalty_delay > 1:
            self.current_clique = {max(self.current_clique, key=lambda v: self.penalties[v])}
        else:
            self.current_clique = {random.choice(range(graph.num_nodes))}
        
        self.step_count += 1
        return self.best_clique
