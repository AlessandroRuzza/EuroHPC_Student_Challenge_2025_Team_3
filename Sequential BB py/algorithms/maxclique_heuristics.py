from abc import ABC, abstractmethod
from collections import defaultdict

import random

class MaxCliqueHeuristic(ABC):
    @abstractmethod
    def find_max_clique(self, graph, union_find, added_edges):
        """
        Returns a list of nodes that form the maximum clique found
        """
        pass



class GreedyCliqueFinder(MaxCliqueHeuristic):
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
    def __init__(self, max_steps=100, penalty_delay=1):
        self.max_steps = max_steps
        self.penalty_delay = penalty_delay

    def find_max_clique(self, graph, union_find, added_edges):
        """
        Dynamic Local Search (DLS) implementation for finding maximum clique
        Returns the size of maximum clique found
        """

        #### Helper functions

        def select_min_penalty(vertices, penalties):
            """Select the vertex with the minimum penalty."""
            min_penalty = min(penalties[v] for v in vertices)
            return random.choice([v for v in vertices if penalties[v] == min_penalty])
        
        def expand(clique, penalties):
            """Expand the current clique by adding suitable vertices."""
            while True:
                candidates = [v for v in range(graph.num_nodes) 
                            if v not in clique and 
                            all(graph.is_connected(v, u) for u in clique)]
                if not candidates:
                    break
                v = select_min_penalty(candidates, penalties)
                clique.add(v)
            return clique
        
        def plateau_search(clique, penalties):
            """Swap vertices to maintain clique size while exploring plateau."""

            steps = 0
            while True and steps < self.max_steps:
                candidates = [v for v in range(graph.num_nodes) 
                            if v not in clique and 
                            sum(graph.is_connected(v, u) for u in clique) == len(clique) - 1]
                if not candidates:
                    break
                v = select_min_penalty(candidates, penalties)
                u = next(u for u in clique if not graph.is_connected(v, u))
                clique.remove(u)
                clique.add(v)
                steps += 1
            return clique

        #### Main search

        # Initialize penalties and clique
        penalties = {v: 0 for v in range(graph.num_nodes)}
        clique = {random.choice(range(graph.num_nodes))}
        best_clique = set(clique)
        
        # Main search loop
        for step in range(self.max_steps):
            expand(clique, penalties)
            plateau_search(clique, penalties)
            
            if len(clique) > len(best_clique):
                best_clique = set(clique)
            
            # Update penalties
            for v in clique:
                penalties[v] += 1
            if step % self.penalty_delay == 0:
                for v in penalties:
                    penalties[v] = max(0, penalties[v] - 1)
            
            # Perturbation to avoid stagnation
            if self.penalty_delay > 1:
                clique = {max(clique, key=lambda v: penalties[v])}
            else:
                v = random.choice(range(graph.num_nodes))
                clique = {v} | {u for u in clique if graph.is_connected(v, u)}
        
        return best_clique