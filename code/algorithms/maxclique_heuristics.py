from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from copy import deepcopy
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



class DLS(MaxCliqueHeuristic):
    """Normal DLS implementation"""

    def __init__(self, max_steps=100, penalty_delay=1):
        """
        Constructor for the DLS class

        :param max_steps: Number of iterations to run the algorithm before stopping
        :type max_steps: int
        :default max_steps: 100
        :param penalty_delay: Delay in penalty updates
        :type penalty_delay: int
        :default penalty_delay: 1
        """
        ## @var max_steps
        # Number of iterations to run the algorithm before stopping
        self.max_steps = max_steps
        ## @var penalty_delay
        # Delay in penalty updates
        self.penalty_delay = penalty_delay
        ## @var penalties
        # Dictionary of penalties for each vertex
        self.penalties = {}
        ## @var current_clique
        # Current clique
        self.current_clique = None
        ## @var best_clique
        # Best clique found
        self.best_clique = None
        ## @var step_count
        # Number of steps taken
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

        :param graph: The graph to expand the clique in
        :type graph: Graph
        :param clique: Current clique
        :type clique: set
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

        :param graph: The graph to plateau search in
        :type graph: Graph
        :param clique: Current clique
        :type clique: set
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

class DLSwithColors(DLS):
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

        :param graph: Graph to expand the clique in 
        :type graph: Graph
        :param clique: Current clique
        :type clique: set
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
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

        :param graph: Graph to expand the clique in 
        :type graph: Graph
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



class DLSAdaptive(DLS):
    """
    Adaptive DLS implementation that switches between basic and color-aware strategies based on the state of the coloring and search progress.
    """
    def __init__(self, max_steps=100, penalty_delay=1, color_threshold=0.75):
        """
        Constructor for the DLSAdaptive class

        :param max_steps: Number of iterations to run the algorithm before stopping
        :type max_steps: int
        :param penalty_delay: Delay in penalty updates
        :type penalty_delay: int
        :param color_threshold: Threshold for switching between strategies (values between 0 and 1)
        :type color_threshold: float
        """
        super().__init__(max_steps, penalty_delay)
        ## @var color_threshold
        # Threshold for switching between strategies
        self.color_threshold = color_threshold
        ## @var basic_dls
        # DLS instance for early stage
        self.basic_dls = DLS(max_steps, penalty_delay)
        ## @var color_dls
        # DLS instance for later stage
        self.color_dls = DLSwithColors(max_steps, penalty_delay)
        ## @var current_strategy
        # Current strategy in use
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


class DLSIncreasingPenalty(DLS):
    """DLS implementation with increasing penalty delay
    After increase_interval steps without improvement, the penalty delay is increased by 1,
    saturating at max_penalty_delay
    """
    def __init__(self, max_steps=100, penalty_delay=1, increase_interval=10, max_penalty_delay=5):
        """
        Constructor for the DLSIncreasingPenalty class

        :param max_steps: Number of iterations to run the algorithm before stopping
        :type max_steps: int
        :param penalty_delay: Delay in penalty updates
        :type penalty_delay: int
        :param increase_interval: Interval for increasing penalty delay
        :type increase_interval: int
        :param max_penalty_delay: Maximum penalty delay
        :type max_penalty_delay: int
        """
        super().__init__(max_steps, penalty_delay)
        ## @var increase_interval
        # Interval for increasing penalty delay
        self.increase_interval = increase_interval
        ## @var max_penalty_delay
        # Maximum penalty delay
        self.max_penalty_delay = max_penalty_delay
        ## @var steps_without_improvement
        # Number of steps without improvement
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


class ParallelDLS(DLS):
    """
    Parallel DLS implementation using multiprocessing to run multiple DLS solvers with different parameters sampled from a Poisson distribution.
    """
    def __init__(self, num_workers=5, lambda_max_steps=10, lambda_penalty_delay=1, dls_instance=DLS()):
        """
        Constructor for the ParallelDLS class

        :param num_workers: Number of parallel DLS solvers to run
        :type num_workers: int
        :param lambda_max_steps: Lambda parameter for Poisson distribution of max_steps
        :type lambda_max_steps: float
        :param lambda_penalty_delay: Lambda parameter for Poisson distribution of penalty_delay
        :type lambda_penalty_delay: float
        :param dls_instance: An instance of the DLS class to use for solving
        :type dls_instance: DLS
        """

        super().__init__(lambda_max_steps, lambda_penalty_delay)
        ## @var num_workers
        # Number of parallel DLS solvers to run
        self.num_workers = num_workers
        ## @var lambda_max_steps
        # Lambda parameter for Poisson distribution of max_steps
        self.lambda_max_steps = lambda_max_steps
        ## @var lambda_penalty_delay
        # Lambda parameter for Poisson distribution of penalty_delay
        self.lambda_penalty_delay = lambda_penalty_delay
        ## @var dls_instance
        # An instance of the DLS class to use for solving
        self.dls_instance = dls_instance
        ## @var best_clique
        # Best clique found across all solvers
        self.best_clique = []
        ## @var current_clique
        # Current clique found by the solvers
        self.current_clique = []




    def single_step(self, graph, union_find, added_edges):

        """Run the parallel DLS solvers and return the best clique found.

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to be added to the graph
        :type added_edges: list
        :return: The best clique found across all solvers
        :rtype: set
        """

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            results = pool.map(self.run_single_solver, 
                               [graph] * self.num_workers, 
                               [union_find]*self.num_workers, 
                               [added_edges]*self.num_workers)
        
        # Return the best clique found
        self.current_clique = max(results, key=len)
        if len(self.current_clique) > len(self.best_clique):
            self.best_clique = set(self.current_clique)
        
        return self.best_clique
        

    def run_single_solver(self, graph, union_find, added_edges):
        """Run a single DLS solver with parameters sampled from Poisson distribution.

        :param graph: The graph to find the maximum clique in
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: List of edges to be added to the graph
        :type added_edges: list
        :return: The best clique found by this solver
        :rtype: set
        """
        worker_max_steps = max(int(np.random.poisson(self.lambda_max_steps)),1)
        worker_penalty_delay = max(int(np.random.poisson(self.lambda_penalty_delay)), 1)
        
        solver = deepcopy(self.dls_instance)
        solver.max_steps = worker_max_steps
        solver.penalty_delay = worker_penalty_delay

        return solver.find_max_clique(graph, union_find, added_edges)