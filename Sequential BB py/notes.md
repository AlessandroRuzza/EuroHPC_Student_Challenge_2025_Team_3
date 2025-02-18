# Strengths
1. The code correctly applies branch-and-bound, using the maximum clique heuristic as a lower bound and the graph coloring heuristic as an upper bound.
2. The implementations of max clique and DSATUR-based coloring provide reasonable estimates for bounds.
3. Using a min-heap (heapq) prioritizes nodes with smaller upper bounds, helping the search converge faster.
4. The Union-Find (Disjoint Set) structure allows efficient merging of nodes when assuming they belong to the same color class.

# Issues and Optimizations
1. Inefficient Maximum Clique Heuristic
- DLS can be improved using information from coloring. (e.g. DLSwithColors)
- The MaxClique heuristic tends to be slow, especially when considering colors.
- Bron-Kerbosch algorithm or an improved greedy heuristic like the pivot strategy to improve lower bounds.
  
2. Improve DSATUR Coloring Heuristic
- The DSATUR heuristic is performant, but other techniques may provide a better approx of chromatic number.
- E.g Fractional coloring

3. Integrate the two heuristics
- If executing MaxClique and Coloring sequentially, we can use information from the first heuristic
- E.g. if DSatur knows a clique, it can already assign different colors to all its elements
- If executing MaxClique and Coloring in parallel, we can use the heuristic results from the parent node 

4. Non-Optimal Branching Strategy
- In the code, pairs are selected prioritizing high-saturation nodes (tiebreak by high-degree) for branching. 
- This branching strategy maximizes upper-bound decrease.
- Instead: choose pairs that maximize lower-bound increase? (heuristic-driven branching).

5. Parallelization Using MPI
- The implementation is sequential. Since each branch is independent:
- MPI for inter-node parallelization (distribute branch searches across processes).
- OpenMP for intra-node optimizations, e.g., computing clique/coloring in parallel.
 
6. Memory Optimization
- zhe deepcopy() calls are expensive. Instead:
- Persistent Union-Find structures (path compression).
- Maintaining edge additions separately instead of copying entire sets.