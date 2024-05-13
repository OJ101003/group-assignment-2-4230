def floyd_warshall(num_nodes, edges):
    # Step 2: Create the initial adjacency matrix
    inf = float('inf')
    dist = [[inf] * num_nodes for _ in range(num_nodes)]
    
    # Step 3: Populate the matrix with edges
    for i in range(num_nodes):
        dist[i][i] = 0  # Distance to itself is 0
    for u, v in edges:
        dist[u][v] = 1  # Distance for an unweighted edge is 1
        dist[v][u] = 1  # Set distance for the reverse since it's undirected
    
    # Step 4: Apply the Floyd-Warshall Algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist
    
edges = [(0, 1), (1, 2), (2, 3),(3,0)]


# Example usage
max_node = max(max(u, v) for u, v in edges) + 1  # Plus one because node indices start from 0
print(max_node)
matrix = floyd_warshall(max_node, edges)


# Printing the matrix
for row in matrix:
    print(row)


        # counter = 0
        # # with open("facebook_combined.txt", 'r') as f:
        # #     for line in f:
        # #         if counter >= 10:
        # #             break
        # #         counter += 1
        # #         parts = line.split()
        # #         if len(parts) == 2:
        # #             edges.append((int(parts[0]), int(parts[1])))