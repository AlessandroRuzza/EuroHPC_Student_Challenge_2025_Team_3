from abc import ABC, abstractmethod
from collections import defaultdict, deque

class BranchingStrategy(ABC):
    """
    Abstract base class for branching strategies
    """
    def find_pair(self, graph, union_find, added_edges):
        """
        Finds a pair of non-adjacent vertices to branch on

        :param graph: Graph to branch on
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
        :return: Tuple of non-adjacent vertices to branch on, or (None, None) if no such pair exists
        :rtype: tuple
        """
        vertices = self.get_ordered_vertices(graph, union_find)
        return self.select_pair(vertices, graph, union_find, added_edges)

    @abstractmethod
    def get_ordered_vertices(self, graph, union_find):
        """
        Order the vertices of the graph according to the strategy

        :param graph: Graph to order vertices of
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :return: List of vertices ordered according to the strategy
        :rtype: list
        """
        pass

    def select_pair(self, vertices, graph, union_find, added_edges):
        """Method to find non-adjacent pairs from ordered vertices and satisfies the color constraints

        :param vertices: List of vertices to choose from
        :type vertices: list
        :param graph: Graph to branch on
        :type graph: Graph
        :param union_find: Data Structure to keep track of vertex colors
        :type union_find: UnionFind
        :return: Tuple of non-adjacent vertices to branch on, or (None, None) if no such pair exists
        :rtype: tuple
        """
        n = len(vertices)
        for i, u in enumerate(vertices):
            for v in vertices[i+1:]:
                ru = union_find.find(u)
                rv = union_find.find(v)
                if ru == rv:
                    continue
                adj = (v in graph.adj_list[u]) # If v,u directly adjacent

                # or (ru, rv) in added_edges or (rv, ru) in added_edges
                # Normal "in" operator doesn't work as we need to look for root of union_find
                for a,b in added_edges:
                    if adj: break
                    u_v_in = union_find.find(a) == ru and union_find.find(b) == rv
                    v_u_in = union_find.find(a) == rv and union_find.find(b) == ru
                    adj = u_v_in or v_u_in
                    
                if not adj:
                    return u, v
        return None, None

class SimpleBranchingStrategy(BranchingStrategy):
    """
    Simple strategy that uses natural ordering
    """
    def get_ordered_vertices(self, graph, union_find):
        return list(range(len(graph)))

class DegreeBranchingStrategy(BranchingStrategy):
    """
    Strategy that orders vertices by degree
    """
    def get_ordered_vertices(self, graph, union_find):
        return sorted(range(len(graph)), 
                     key=lambda x: graph.degree(x), 
                     reverse=True)

class SaturationBranchingStrategy(BranchingStrategy):
    """
    Strategy that orders vertices by saturation degree
    """
    def get_ordered_vertices(self, graph, union_find):
        def saturation(vertex):
            """
            Calculate the saturation degree of a vertex

            :param vertex: Vertex to calculate saturation degree of
            :type vertex: int
            :return: Saturation degree of the vertex
            :rtype: int
            """
            neighbor_colors = set()
            for neighbor in graph.adj_list[vertex]:
                neighbor_colors.add(union_find.find(neighbor))
            return len(neighbor_colors)
        
        return sorted(range(len(graph)), 
                     key=lambda x: (saturation(x), graph.degree(x)), 
                     reverse=True)
