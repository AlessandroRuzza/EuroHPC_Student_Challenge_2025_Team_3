from graph import Graph


# Generate a random graph with num_nodes nodes and density probability of edge between any two nodes
def generate_random_graph(num_nodes, density):
    graph = Graph(num_nodes)
    random.seed(time.time())

    for i in range(1, num_nodes):
        for j in range(i):
            if random.random() < density:
                graph.add_edge(i, j)

    return graph


# Parse a .col file and return a graph
def parse_col_file(file_path):
    graph = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("c "):
                continue  # Ignore comments

            if line.startswith("p "):
                parts = line.split()
                if len(parts) == 4 and parts[0] == "p" and parts[1] == "edge":
                    num_nodes = int(parts[2])
                    graph = Graph(num_nodes)
                continue

            if line.startswith("e "):
                parts = line.split()
                if len(parts) == 3:
                    node1, node2 = int(parts[1])-1, int(parts[2])-1 # files count nodes from 1, we count from 0
                    graph.add_edge(node1, node2)
                continue

    return graph