from abc import ABC, abstractmethod
from collections import defaultdict, deque

class BranchingStrategy(ABC):
    """
    Abstract base class for branching strategies
    """
    def find_pair(self, graph, union_find, added_edges, depth):
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
        vertices = self.get_ordered_vertices(graph, union_find, depth)
        return self.select_pair(vertices, graph, union_find, added_edges)

    @abstractmethod
    def get_ordered_vertices(self, graph, union_find, depth):
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
        :param added_edges: Data structure to keep track of vertices with different colors
        :type added_edges: list
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
    def get_ordered_vertices(self, graph, union_find, depth):
        return list(range(len(graph)))

class DegreeBranchingStrategy(BranchingStrategy):
    """
    Strategy that orders vertices by degree
    """
    def get_ordered_vertices(self, graph, union_find, depth):
        return sorted(range(len(graph)), 
                     key=lambda x: graph.degree(x), 
                     reverse=True)

class SaturationBranchingStrategy(BranchingStrategy):
    """
    Strategy that orders vertices by saturation degree
    """
    def get_ordered_vertices(self, graph, union_find, depth):
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


class HybridBranching(BranchingStrategy):
    def __init__(self):
        self.depth_thresholds = {"early": 5, "mid": 15}  # Define heuristic switching points
        self.log = []  # Initialize a log list to store branching information

    def get_ordered_vertices(self, graph, union_find, depth):
        """Dynamically select the best branching heuristic based on search depth."""
        
        if depth < self.depth_thresholds["early"]:
            # Early: Clique-based selection (hardest constraints first)
            clique = graph.find_max_clique(union_find, set())
            self.log.append(f"Early stage: Using clique-based selection at depth {depth}\n")
            self.log.append(f"Max clique found: {clique}\n")
            return sorted(clique, key=lambda x: graph.degree(x), reverse=True)

        elif depth < self.depth_thresholds["mid"]:
            # Mid: Conflict-driven selection (focus on high-conflict nodes)
            self.log.append(f"Mid stage: Using conflict-driven selection at depth {depth}\n")
            return sorted(
                range(len(graph)),
                key=lambda x: (graph.degree(x), self.saturation(graph, union_find, x)),
                reverse=True
            )

        else:
            # Late: DSATUR-guided selection (lowest available color saturation)
            self.log.append(f"Late stage: Using DSATUR-guided selection at depth {depth}\n")
            return sorted(
                range(len(graph)),
                key=lambda x: self.saturation(graph, union_find, x),
                reverse=False  # Prefer nodes with lower saturation
            )

    def saturation(self, graph, union_find, node):
        """Compute saturation: Number of unique colors among node’s neighbors."""
        saturation_value = len({union_find.find(neigh) for neigh in graph.adj_list[node]})
        self.log.append(f"Node {node} has saturation value: {saturation_value}\n")
        return saturation_value
