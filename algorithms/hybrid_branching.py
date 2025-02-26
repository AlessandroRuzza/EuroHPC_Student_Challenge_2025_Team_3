from copy import deepcopy
from branching_strategies import BranchingStrategy
from graph.base import BranchAndBoundNode

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

    def branch_node(self, graph, node):
        """
        Computing the children nodes of a given node in the branch and bound tree with its upper and lower bounds

        :param graph: graph to solve
        :type graph: Graph
        :param node: node to branch on
        :type node: BranchAndBoundNode
        :return: list of children nodes, one with the same color for the pair of nodes and one with different colors
        :rtype: list[BranchAndBoundNode]
        """

        # Find a pair of nodes to branch on
        u, v = graph.find_pair(node.union_find, node.added_edges)
        if u is None:
            return []
        
        childNodes = []

        # Branch 1: Same color
        color_u = node.union_find.find(u)
        color_v = node.union_find.find(v)
        doBranch1 = True
        
        # Do not create branch if assignment is already invalid
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

            clique = graph.find_max_clique(uf1, edges1)
            lb1 = len(clique)
            
            coloring = graph.find_coloring(uf1, edges1)
            ub1 = len(set(coloring))

            childNodes.append(BranchAndBoundNode(uf1, edges1, lb1, ub1, coloring))
            self.log.append(f"Node {node.id} branched by imposing {u}, {v} have the same color \n")
            self.log.append(f"Branch 1 child node results: \n")
            self.log.append(f"Clique (LB = {lb1}) = {clique}\n")
            self.log.append(f"Coloring (UB = {ub1}) = {coloring}\n\n")

        # Branch 2: Different color
        uf2 = deepcopy(node.union_find)
        edges2 = deepcopy(node.added_edges)
        ru = uf2.find(u)
        rv = uf2.find(v)
        edges2.add((ru, rv))

        clique = graph.find_max_clique(uf2, edges2)
        lb2 = len(clique)
        coloring = graph.find_coloring(uf2, edges2)
        ub2 = len(set(coloring))

        childNodes.append(BranchAndBoundNode(uf2, edges2, lb2, ub2, coloring))

        self.log.append(f"Node {node.id} branched by imposing vertices {u}, {v} have different colors\n")
        self.log.append(f"Branch 2 child node results: \n")
        self.log.append(f"Clique (LB = {lb2}) = {clique}\n")
        self.log.append(f"Coloring (UB = {ub2}) = {coloring}\n\n")

        return childNodes