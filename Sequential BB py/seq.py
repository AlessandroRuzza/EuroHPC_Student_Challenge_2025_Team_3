import heapq
from copy import deepcopy

# lb - lower bound
# ub - upper bound

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

class BranchAndBoundNode:
    def __init__(self, union_find, added_edges, lb, ub):
        self.union_find = union_find
        self.added_edges = set(added_edges)
        self.lb = lb
        self.ub = ub

    def __lt__(self, other):
        return self.ub < other.ub

def heuristic_max_clique(graph, union_find, added_edges):
    # Simplified greedy max clique heuristic
    merged = {}
    
    for u in range(len(graph)):
        root = union_find.find(u)
        if root not in merged:
            merged[root] = set()
        for v in graph[u]:
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
    return len(clique)

def heuristic_coloring(graph, union_find, added_edges):
    # Simplified DSATUR heuristic
    roots = {union_find.find(u) for u in range(len(graph))}
    merged_adj = {r: set() for r in roots}
    
    for u in range(len(graph)):
        ru = union_find.find(u)
        for v in graph[u]:
            rv = union_find.find(v)
            if ru != rv:
                merged_adj[ru].add(rv)
        for a, b in added_edges:
            ra = union_find.find(a)
            rb = union_find.find(b)
            if ra == ru and rb != ru:
                merged_adj[ru].add(rb)
            if rb == ru and ra != ru:
                merged_adj[ru].add(ra)
                
    color = {}
    
    for node in sorted(merged_adj, key=lambda x: -len(merged_adj[x])):
        used = {color[n] for n in merged_adj[node] if n in color}
        c = 0
        while c in used:
            c += 1
        color[node] = c
    return max(color.values()) + 1 if color else 0

def branch_and_bound(graph, time_limit=10000):
    n = len(graph)
    initial_uf = UnionFind(n)
    initial_edges = set()
    
    lb = heuristic_max_clique(graph, initial_uf, initial_edges)
    ub = heuristic_coloring(graph, initial_uf, initial_edges)
    
    best_ub = ub
    queue = []
    heapq.heappush(queue, (ub, BranchAndBoundNode(initial_uf, initial_edges, lb, ub)))
    
    while queue:
        current_ub, node = heapq.heappop(queue)
        if node.lb >= best_ub:
            continue
        if current_ub < best_ub:
            best_ub = current_ub
        if node.lb == best_ub:
            break
        
        # Non-adjacent pair
        found = False
        for u in range(n):
            for v in range(u+1, n):
                ru = node.union_find.find(u)
                rv = node.union_find.find(v)
                if ru == rv:
                    continue
                adj = (rv in graph[ru]) or ((ru, rv) in node.added_edges) or ((rv, ru) in node.added_edges)
                if not adj:
                    found = True
                    break
            if found:
                break
        
        if not found:
            continue
        
        # Branch 1: same color
        uf1 = deepcopy(node.union_find)
        uf1.union(u, v)
        edges1 = deepcopy(node.added_edges)
        lb1 = heuristic_max_clique(graph, uf1, edges1)
        ub1 = heuristic_coloring(graph, uf1, edges1)
        if lb1 < best_ub:
            heapq.heappush(queue, (ub1, BranchAndBoundNode(uf1, edges1, lb1, ub1)))
        
        # Branch 2: different color
        uf2 = deepcopy(node.union_find)
        edges2 = deepcopy(node.added_edges)
        ru = uf2.find(u)
        rv = uf2.find(v)
        if (ru, rv) not in edges2 and (rv, ru) not in edges2:
            edges2.add((ru, rv))
        lb2 = heuristic_max_clique(graph, uf2, edges2)
        ub2 = heuristic_coloring(graph, uf2, edges2)
        if lb2 < best_ub:
            heapq.heappush(queue, (ub2, BranchAndBoundNode(uf2, edges2, lb2, ub2)))
    
    return best_ub