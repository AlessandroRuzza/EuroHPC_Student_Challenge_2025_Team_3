from abc import ABC, abstractmethod
from collections import defaultdict, deque

class BranchingStrategy(ABC):
    def find_pair(self, graph, union_find, added_edges):
        """
        Returns a tuple (u, v) of non-adjacent vertices to branch on,
        or (None, None) if no such pair exists
        """
        vertices = self.get_ordered_vertices(graph, union_find)
        return self.select_pair(vertices, graph, union_find, added_edges)

    @abstractmethod
    def get_ordered_vertices(self, graph, union_find):
        """
        Returns vertices ordered according to the strategy
        """
        pass

    def select_pair(self, vertices, graph, union_find, added_edges):
        """Method to find non-adjacent pairs from ordered vertices"""
        n = len(vertices)
        for i, u in enumerate(vertices):
            for v in vertices[i+1:]:
                ru = union_find.find(u)
                rv = union_find.find(v)
                if ru == rv:
                    continue
                adj = (v in graph.adj_list[u]) or ((ru, rv) in added_edges) or ((rv, ru) in added_edges)
                if not adj:
                    return u, v
        return None, None

class DefaultBranchingStrategy(BranchingStrategy):
    def get_ordered_vertices(self, graph, union_find):
        """Simple strategy that uses natural ordering"""
        return list(range(len(graph)))

class DegreeBranchingStrategy(BranchingStrategy):
    def get_ordered_vertices(self, graph, union_find):
        """Returns vertices ordered by degree"""
        return sorted(range(len(graph)), 
                     key=lambda x: graph.degree(x), 
                     reverse=True)

class SaturationBranchingStrategy(BranchingStrategy):
    def get_ordered_vertices(self, graph, union_find):
        """Returns vertices ordered by saturation degree"""

        # Helper function to calculate saturation degree
        def saturation(vertex):
            neighbor_colors = set()
            for neighbor in graph.adj_list[vertex]:
                neighbor_colors.add(union_find.find(neighbor))
            return len(neighbor_colors)
        
        return sorted(range(len(graph)), 
                     key=lambda x: (saturation(x), graph.degree(x)), 
                     reverse=True)
